import unittest
from pathlib import Path
from tests.context_test_support import safe_temporary_directory

from workflows.sbi.e2e_vdm_context_interactive import parse_accounting,request_size,allocation_command,child_command
from workflows.sbi.e2e_vdm_context_queue import task_lock,disk_bytes
from workflows.sbi.e2e_vdm_context_data import spec


class InteractiveTests(unittest.TestCase):
    def test_real_allocated_accounting_includes_idle_gpus(self):
        value=parse_accounting('1|COMPLETED|0:0|1800|cpu=64,mem=20G|\n'
            '2|FAILED|75:0|3600|cpu=128,gres/gpu=4,mem=256G|\n',{'1':0,'2':4})
        self.assertEqual(value['gpu_hours'],4)
        self.assertEqual(value['cpu_node_hours'],.5)
        for text,expected in [('1|RUNNING|0:0|20|cpu=64|',{'1':0}),('',{'1':0})]:
            with self.assertRaises(RuntimeError):
                parse_accounting(text,expected)
        with self.assertRaises(ValueError):
            parse_accounting('2|COMPLETED|0:0|3600|gres/gpu=4|',{'2':1})

    def test_request_limits_and_tail_resource_size(self):
        budget=spec()['budget']
        usage=dict(gpu_hours=0,cpu_node_hours=1.64)
        self.assertEqual(request_size('train',10,usage,budget,200000,0),(4,240))
        self.assertEqual(request_size('all',1,usage,budget,200000,0),(1,240))
        self.assertEqual(request_size('all',2,usage,budget,200000,0),(2,240))
        self.assertEqual(request_size('smoke',1,usage,budget,200000,0),(1,60))
        self.assertEqual(request_size('report',1,usage,budget,200000,0),(0,120))
        self.assertEqual(request_size('all',10,dict(usage,gpu_hours=110),budget,200000,0),(4,30))
        with self.assertRaises(RuntimeError):
            request_size('all',10,dict(usage,gpu_hours=112),budget,200000,0)
        with self.assertRaises(RuntimeError):
            request_size('train',10,usage,budget,1000,0)

    def test_commands_keep_compute_inside_explicit_allocation(self):
        root=Path('/tmp/vdm')
        for gpus,qos in ((1,'shared_interactive'),(2,'shared_interactive'),(4,'interactive'),(0,'interactive')):
            cmd=allocation_command(root,0,gpus,30)
            self.assertEqual(cmd[0],'salloc')
            self.assertIn('--qos='+qos,cmd)
            self.assertIn('--licenses=scratch',cmd)
            self.assertIn('--immediate=600',cmd)
            self.assertIn('--time=00:30:00',cmd)
            self.assertEqual('--gpus='+str(gpus) in cmd,bool(gpus))
        cmd=child_command(root,'train',dict(arm='D',seed=1,factor='coarse'))
        self.assertIn('workflows.sbi.e2e_vdm_context_train',cmd)
        self.assertNotIn('--restart-test',cmd)

    def test_queue_exclusion_and_bounded_storage_count(self):
        with safe_temporary_directory() as tmp:
            root=Path(tmp)
            with task_lock(root/'queue') as first:
                self.assertTrue(first)
                with task_lock(root/'queue') as second:
                    self.assertFalse(second)
            with task_lock(root/'queue') as third:
                self.assertTrue(third)
            with (root/'payload').open('xb') as stream:
                stream.write(b'12345')
            (root/'alias').hardlink_to(root/'payload')
            self.assertEqual(disk_bytes(root),5)


if __name__=='__main__':
    unittest.main()
