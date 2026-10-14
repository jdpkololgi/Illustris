"""One Gaussian-compatible learned exterior-conditioned CFM pilot.

The velocity is neural in time and affine in state/context. This restricted
architecture is intentionally appropriate for Gaussian targets and permits
exact post-training moment checks; it is not the cosmological 3D network.
"""
import copy
import math

import numpy as np
from scipy.linalg import solve_triangular
import torch
from torch import nn

import e2e_gaussian_block_sampler as block
import e2e_restricted_context as context


class GaussianTrainingData:
    """Joint prior/likelihood samples; analytic conditionals are QA ONLY."""
    def __init__(self,n=4,kind='long_range',regime=1):
        self.n,self.kind,self.regime = n,kind,regime
        self.prior = context.prior(n,kind)
        self.mask,self.std = context.observations(n,regime)
        self.observed = np.flatnonzero(self.mask)
        self.patches = np.stack(context.patches(n))
        self.size = n**3
        self.width = self.patches.shape[1]
        self.exteriors = np.stack([np.setdiff1d(np.arange(self.size),ix) for ix in self.patches])
        self.context_dimensions = self.exteriors.shape[1]+len(self.observed)
        observed_cov = self.prior[np.ix_(self.observed,self.observed)]+np.diag(self.std[self.observed]**2)
        joint = np.block([[self.prior,self.prior[:,self.observed]],
                          [self.prior[self.observed],observed_cov]])
        transforms,responses,covariances = [],[],[]
        for indices,exterior in zip(self.patches,self.exteriors):
            columns = np.concatenate([exterior,self.size+np.arange(len(self.observed))])
            covariance = joint[np.ix_(columns,columns)]
            white = solve_triangular(np.linalg.cholesky(covariance),np.eye(len(columns)),lower=True)
            transform = np.zeros((len(columns),len(joint)))
            transform[:,columns] = white
            np.testing.assert_allclose(transform@joint@transform.T,np.eye(len(columns)),atol=1e-11)
            # Kept exclusively for evaluation; never supplied to train_step.
            response = joint[np.ix_(indices,columns)]@white.T
            residual = block.symmetric(self.prior[np.ix_(indices,indices)]-response@response.T)
            np.linalg.cholesky(residual)
            transforms.append(transform); responses.append(response); covariances.append(residual)
        self.transform = np.stack(transforms)
        self.exact_response = np.stack(responses)
        self.exact_covariance = np.stack(covariances)
        self.prior_factor = np.linalg.cholesky(self.prior)
        self._device_cache = {}

    def tensor_data(self,device,dtype):
        key = (str(device),dtype)
        if key not in self._device_cache:
            tensor = lambda x:torch.as_tensor(x,device=device,dtype=dtype)
            self._device_cache[key] = dict(factor=tensor(self.prior_factor),transform=tensor(self.transform),
                std=tensor(self.std[self.observed]),
                patches=torch.as_tensor(self.patches,device=device,dtype=torch.long),
                observed=torch.as_tensor(self.observed,device=device,dtype=torch.long))
        return self._device_cache[key]

    def batch(self,count,device,generator,dtype=torch.float32):
        data = self.tensor_data(device,dtype)
        field = torch.randn((count,self.size),device=device,dtype=dtype,generator=generator)@data['factor'].T
        y = field[:,data['observed']]+data['std']*torch.randn((count,len(self.observed)),
            device=device,dtype=dtype,generator=generator)
        raw = torch.cat([field,y],dim=-1)
        conditions = torch.einsum('rj,bcj->rbc',raw,data['transform'])
        return field[:,data['patches']],conditions

    def heldout(self,count=8,seed=61019500):
        result = []
        for index in range(count):
            result.append(context.problem(self.n,self.kind,self.regime,seed+index))
        return result


class TimeAffineCFM(nn.Module):
    def __init__(self,blocks,width,context_dimensions,hidden=128):
        super().__init__()
        self.blocks,self.width,self.context_dimensions = blocks,width,context_dimensions
        self.inputs = width+context_dimensions+1
        self.network = nn.Sequential(nn.Linear(5,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),
            nn.Linear(hidden,blocks*width*self.inputs))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def matrices(self,t):
        features = torch.stack([t,torch.sin(math.pi*t),torch.cos(math.pi*t),
                                torch.sin(2*math.pi*t),torch.cos(2*math.pi*t)],dim=-1)
        return self.network(features).reshape(len(t),self.blocks,self.width,self.inputs)

    def forward(self,state,t,conditions):
        """state/context shapes: time-group, realization, block, coordinate."""
        if state.ndim != 4 or state.shape[:3] != conditions.shape[:3] or len(t)!=len(state):
            raise ValueError('Expected time-group by realization by block tensors')
        features = torch.cat([state,conditions,torch.ones_like(state[...,:1])],dim=-1)
        return torch.einsum('trbi,tboi->trbo',features,self.matrices(t))


def train_step(model,optimizer,data,generator,batch=2048,time_groups=32):
    if batch % time_groups:
        raise ValueError('Batch must contain equal time groups')
    device = next(model.parameters()).device
    endpoint,conditions = data.batch(batch,device,generator)
    shape = (time_groups,batch//time_groups,len(data.patches),data.width)
    endpoint = endpoint.reshape(shape)
    conditions = conditions.reshape(*shape[:3],data.context_dimensions)
    noise = torch.randn(shape,device=device,generator=generator)
    t = (torch.arange(time_groups,device=device)+torch.rand(time_groups,device=device,generator=generator))/time_groups
    tt = t[:,None,None,None]
    state = (1-tt)*noise+tt*endpoint
    prediction = model(state,t,conditions)
    loss = (prediction-(endpoint-noise)).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite CFM training loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def heun_affine(coefficients,width):
    """Affine propagation through the SAME two-evaluation Heun update."""
    steps = len(coefficients)-1
    if steps < 1:
        raise ValueError('At least one integration step required')
    count = coefficients.shape[1]
    identity = np.broadcast_to(np.eye(width),(count,width,width)).copy()
    initial = identity.copy()
    conditions = np.zeros((count,width,coefficients.shape[-1]-width))
    dt = 1/steps
    for first,second in zip(coefficients[:-1],coefficients[1:]):
        a,b = first[...,:width],first[...,width:]
        aa,bb = second[...,:width],second[...,width:]
        matrix = identity+dt/2*(a+aa@(identity+dt*a))
        offset = dt/2*(b+dt*(aa@b)+bb)
        initial = matrix@initial
        conditions = matrix@conditions+offset
    return initial,conditions


def exact_coefficients(data,nfe):
    if nfe < 2 or nfe % 2:
        raise ValueError('Positive even NFE required')
    rows = []; identity = np.eye(data.width)
    values,vectors = np.linalg.eigh(data.exact_covariance)
    for t in np.linspace(0,1,nfe//2+1):
        eigen = (t*values-(1-t))/(t*t*values+(1-t)**2)
        a = (vectors*eigen[:,None,:])@vectors.transpose(0,2,1)
        response = (identity-t*a)@data.exact_response
        rows.append(np.concatenate([a,response,np.zeros((len(a),data.width,1))],axis=-1))
    return np.stack(rows)


@torch.no_grad()
def learned_coefficients(model,nfe):
    if nfe < 2 or nfe % 2:
        raise ValueError('Positive even NFE required')
    copy_model = copy.deepcopy(model).cpu().double().eval()
    times = torch.linspace(0,1,nfe//2+1,dtype=torch.float64)
    return copy_model.matrices(times).numpy()


def kernels(data,endpoint,case):
    noise,condition = endpoint
    raw = condition[...,:-1]@data.transform
    y = case['y'][data.observed]
    updates = []
    for i,indices in enumerate(data.patches):
        gain = raw[i,:,:data.size]
        np.testing.assert_allclose(gain[:,indices],0,atol=1e-12)
        offset = condition[i,:,-1]+raw[i,:,data.size:]@y
        covariance = block.symmetric(noise[i]@noise[i].T)
        updates.append(context.GaussianBlock(indices,gain,offset,covariance,case['mu'],len(data.exteriors[i])))
    return updates


def conditional_metrics(data,endpoint):
    noise,response = endpoint
    result = []
    for index in range(len(data.patches)):
        target = data.exact_covariance[index]
        white = solve_triangular(np.linalg.cholesky(target),np.eye(data.width),lower=True)
        covariance = noise[index]@noise[index].T
        difference = block.symmetric(white@(covariance-target)@white.T)
        mean_difference = white@(response[index,:,:-1]-data.exact_response[index])
        mean_bias = white@response[index,:,-1]
        rms = float(np.sqrt((np.sum(mean_difference**2)+np.sum(mean_bias**2))/data.width))
        worst_cov = float(np.max(abs(np.linalg.eigvalsh(difference))))
        result.append(dict(block=index,mean_whitened_rms=rms,
            mean_response_operator_error=float(np.linalg.norm(mean_difference,ord=2)),
            covariance_relative=float(np.linalg.norm(covariance-target)/np.linalg.norm(target)),
            worst_relative_covariance_error=worst_cov,
            component_accuracy_passed=rms<=.01 and worst_cov<=.01))
    return result


@torch.no_grad()
def actual_model_check(model,data,nfe=128):
    """Call the actual neural model inside Heun, independently of matrix propagation."""
    copy_model = copy.deepcopy(model).cpu().double().eval()
    generator = torch.Generator().manual_seed(61019931)
    endpoint,conditions = data.batch(32,'cpu',generator,dtype=torch.float64)
    noise = torch.randn(endpoint.shape,generator=generator,dtype=torch.float64)
    state = noise[None].clone(); condition = conditions[None]
    steps = nfe//2; dt = 1/steps
    for index in range(steps):
        t = torch.tensor([index/steps],dtype=torch.float64)
        tt = torch.tensor([(index+1)/steps],dtype=torch.float64)
        first = copy_model(state,t,condition)
        second = copy_model(state+dt*first,tt,condition)
        state = state+dt/2*(first+second)
    a,b = heun_affine(learned_coefficients(copy_model,nfe),data.width)
    inputs = np.concatenate([conditions.numpy(),np.ones((*conditions.shape[:-1],1))],axis=-1)
    predicted = np.einsum('rbi,boi->rbo',noise.numpy(),a)+np.einsum('rbi,boi->rbo',inputs,b)
    discrepancy = float(np.max(abs(state[0].numpy()-predicted)))
    np.testing.assert_allclose(state[0].numpy(),predicted,atol=2e-10,rtol=2e-10)
    return dict(passed=True,draws=32,nfe=nfe,max_absolute_error=discrepancy)
