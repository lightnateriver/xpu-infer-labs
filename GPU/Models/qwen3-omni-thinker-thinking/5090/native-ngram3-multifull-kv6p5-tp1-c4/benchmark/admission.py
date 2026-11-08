"""FIFO admission using frozen prompt counts and an explicit output reserve."""
from collections import deque
from pathlib import Path
import json,os,threading,time


class TokenGate:
    def __init__(self,root,output):
        self.root=root;self.output=output
        self.ratio=float(os.environ.get('BENCH_ADMISSION_RATIO','0'))
        self.capacity=int(os.environ.get('BENCH_KV_CAPACITY','136528'))
        self.reserve=int(os.environ.get('BENCH_OUTPUT_RESERVE','9000'))
        self.limit=int(self.ratio*self.capacity)
        self.condition=threading.Condition();self.active={};self.events=[];self.order={};self.next=0

    def configure(self,selected):
        self.order={v['sample_id']:i for i,v in enumerate(selected)}

    def acquire(self,item):
        stamp=time.monotonic();identity=item['sample_id']
        prompt=json.loads((self.root/'capacity'/(identity+'.json')).read_text())['input_tokens']
        amount=prompt+self.reserve;assert amount<=self.limit
        with self.condition:
            while self.order[identity]!=self.next or sum(self.active.values())+amount>self.limit:
                self.condition.wait(timeout=1)
            self.active[identity]=amount;self.next+=1
            row=dict(event='admit',epoch=time.time(),sample_id=identity,prompt_tokens=prompt,
                output_reserve=self.reserve,reserved_tokens=amount,active=len(self.active),
                total_reserved=sum(self.active.values()),budget=self.limit,wait_seconds=time.monotonic()-stamp)
            self.events.append(row);self.condition.notify_all();return row

    def release(self,item):
        with self.condition:
            self.active.pop(item['sample_id'])
            self.events.append(dict(event='release',epoch=time.time(),sample_id=item['sample_id'],
                active=len(self.active),total_reserved=sum(self.active.values()),budget=self.limit))
            temp=self.output/'admission-events.tmp';temp.write_text(json.dumps(self.events,indent=2))
            temp.replace(self.output/'admission-events.json');self.condition.notify_all()
