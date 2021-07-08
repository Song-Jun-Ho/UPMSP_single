import simpy
import os
import random
import pandas as pd
import numpy as np
import math
from collections import OrderedDict
import functools

# save_path = './result'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)


# weight = [1, 1.2, 1.4, 1.5, 1.6, 2, 2.1, 2.3, 2.5, 2.6]
# job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# expon_dist = functools.partial(np.random.exponential, size=None)
# process_time = [30, 40, 55, 60, 80, 70, 65, 83, 66, 92]
# process_list = ['LH', 'Sink']
# priority = {'LH': [1, 1, 1, 2, 2, 2, 2, 2]}
# model = {}
# machine 3대 Auto , 5대 Manual

class Part(object):
    def __init__(self, name, type, process_time, weight):
        # 해당 Part 번호
        self.id = name

        # 해당 Part type
        self.type = type

        # average process time
        self.process_time = process_time
        # determined process time
        self.real_proc_time = None
        # machine 별 avg process time
        self.proc_time_Auto = [self.process_time * 0.8 for _ in range(3)]
        self.proc_time_Manual = [self.process_time * 1.2 for _ in range(5)]
        self.p_ij = self.proc_time_Auto + self.proc_time_Manual

        # 작업을 완료한 공정의 수
        self.step = 0

        # part의 납기일 정보
        self.due_date = None

        self.W_ij = None

        self.weight = weight

        self.completion_time = None

    def set_due_date(self, arrival_time):
        self.due_date = arrival_time + 1 * self.process_time


class Source(object):
    def __init__(self, env, IAT, weight, job_type, process_time, model, process_list):
        self.env = env
        self.name = 'Source'

        self.parts_sent = 0
        self.IAT = IAT

        # job type 별 특징
        self.weight = weight
        self.job_type = job_type
        # avg process time
        self.process_time = process_time

        self.action = env.process(self.run())
        self.generated_job_types = np.zeros(len(self.job_type))

        self.model = model
        self.process_list = process_list

    def run(self):
        while True:
            jb_type = np.random.choice(self.job_type)
            self.generated_job_types[jb_type] += 1
            w = self.weight[jb_type]
            p = self.process_time[jb_type]

            # generate job
            part = Part(name='job{0}_{1}'.format(jb_type, self.generated_job_types[jb_type]), type=jb_type,
                        process_time=p, weight=w)
            part.set_due_date(self.env.now)

            # put job to next process buffer_to_machine
            self.model[self.process_list[part.step]].buffer_to_machine.put(part)

            self.model[self.process_list[part.step]].wait_before_new_arrival.succeed()
            self.model[self.process_list[part.step]].wait_before_new_arrival = self.env.event()

            inter_arrival_time = np.random.exponential(self.IAT)
            yield self.env.timeout(inter_arrival_time)

            self.parts_sent += 1

            if self.parts_sent == 1000:
                break


########################################################################################################################

class Process(object):
    def __init__(self, env, name, machine_num, priority, model, process_list, capacity=float('inf'),
                 capa_to_machine=float('inf'), capa_to_process=float('inf')):
        self.env = env
        self.name = name
        self.model = model
        self.machine_num = machine_num

        self.process_list = process_list

        self.capa = capacity
        self.priority = priority

        self.parts_sent = 0
        self.parts_routed = 0

        self.buffer_to_machine = simpy.FilterStore(env, capacity=capa_to_machine)
        self.buffer_to_process = simpy.Store(env, capacity=capa_to_process)

        self.machine_store = simpy.FilterStore(env, capacity=machine_num)

        self.machines = [Machine(env, self.name, 'Machine_{0}'.format(i), idx=i, priority=self.priority[i],
                                 out=self.buffer_to_process, model=model) for i in range(machine_num)]

        for i in range(self.machine_store.capacity):
            self.machine_store.put(self.machines[i])

        # if self.name != 'LH':
        #     env.process(self._to_machine())
        env.process(self._to_machine())
        env.process(self.check_idle_machine())
        env.process(self.check_part_exist())
        env.process(self._to_process())

        # idle machine 이 있을 때 열리고(succeed) 없을 때 닫혀있는 스위치(valve?) event
        # check_idle_machine(Process)에 의해서 제어 됨
        self.idle_machine = env.event()
        self.parts_exist = env.event()
        # idle machine 을 check할 시점이 되면 열리고(succeed) 없을 때 닫혀있는 스위치(valve?) event
        self.wait_before_check = env.event()
        self.wait_before_new_arrival = env.event()

        self.action = 0  # env.step에서는 Process0의 self.action을 매 step마다 선택

        self.working_process_list = dict()  # 현재 작업 진행중인 Process list

    ### Process_1 의 _to_machine process가 agent로 대체되는 것임
    ####################### 이 부분을 많이 수정했습니다.....########################################################
    def _to_machine(self):
        self.i = 0
        step = 0
        while True:
            # print("++++++++++++++++++++ " + self.name + '  {0}th iteration'.format(self.i) + " ++++++++++++++++++++")
            # wait until there exists an idle machine
            self.routing_logic = None
            self.i += 1
            yield self.idle_machine
            yield self.parts_exist
            # If there exist idle machines and also parts in buffer_to_machine
            # Then take action (until one of the items becomes zero)
            if len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                while len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                    self.routing_logic = self.routing(self.action)

                    idle = [x.name for x in self.machine_store.items]
                    # print(str(self.env.now) + '  ' + self.name + '  idle machines : ')
                    # print(idle)
                    # print(str(self.env.now) + ' ' + self.name + '  parts in buffer : ')
                    # print([x.id for x in self.buffer_to_machine.items])

                    part = yield self.buffer_to_machine.get()
                    # print(part)

                    machine = yield self.machine_store.get()

                    ################################ Processing Process generated ##########################
                    self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))

                    self.parts_routed += 1
                # print(self.env.now, '  ', self.name, '  ', 'step{0} Routing finished'.format(step))
                step += 1

            # else:
            #     # If there exist idle machines but no parts in buffer_to_machine...
            #     # You cannot take action yet , you need to wait for a new part arrival...
            #     # 이런 경우는 그냥 SPT를 따르도록 설정해 둠
            #     # Machine의 priority에 의해 (Auto가 idle 시 Auto로 routing 그렇지 않으면 Manu로 routing)
            #
            #     idle = [x.name for x in self.machine_store.items]
            #     # print(str(self.env.now) + '  ' + self.name + '  idle machines : ')
            #     # print(idle)
            #     # print(str(self.env.now) + '  ' + self.name + '  parts in buffer : ')
            #     # print([x.id for x in self.buffer_to_machine.items])
            #
            #     ############## buffer part가 존재하고 동시에 idle machine 이 존재할 때만 돌아가게끔 하기 위한 장치###############
            #     part = yield self.buffer_to_machine.get()
            #     # print(part)
            #     # print(self.env.now, '  ', self.name, '  Now new part arrived')
            #
            #     # If there a new part arrival you can now take action
            #     if self.routing_logic == None:
            #         self.routing_logic = self.routing(self.action)
            #         # print(self.env.now, '   ', self.name, '  Routing finished')
            #
            #     machine = yield self.machine_store.get()
            #
            #     ################################ Processing Process generated ##########################
            #     self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))
            #
            #     self.parts_routed += 1

            # print(str(self.env.now) + ' ' + self.name + '   all available Process_entered')

    def to_machine(self):
        if len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
            while len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                self.routing_logic = self.routing(self.action)

                idle = [x.name for x in self.machine_store.items]

                part = yield self.buffer_to_machine.get()

                machine = yield self.machine_store.get()

                self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))

                self.parts_routed += 1

        else:
            # If there exist idle machines but no parts in buffer_to_machine...
            # You cannot take action yet , you need to wait for a new part arrival...
            # 이런 경우는 그냥 SPT를 따르도록 설정해 둠
            # Machine의 priority에 의해 (Auto가 idle 시 Auto로 routing 그렇지 않으면 Manu로 routing)

            ############## buffer part가 존재하고 동시에 idle machine 이 존재할 때만 돌아가게끔 하기 위한 장치###############
            part = yield self.buffer_to_machine.get()
            # print(part)
            # print(self.env.now, '  ', self.name, '  Now new part arrived')

            # If there a new part arrival you can now take action
            if self.routing_logic == None:
                self.routing_logic = self.routing(self.action)
                # print(self.env.now, '   ', self.name, '  Routing finished')

            machine = yield self.machine_store.get()

            ################################ Processing Process generated ##########################
            self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))

            self.parts_routed += 1

    def check_idle_machine(self):  # idle_machine event(valve)를 제어하는 Process
        while True:
            if len(self.machine_store.items) != 0:
                # idle machine 있을 시 idle_machine event(valve)를 열었다가 바로 닫는다.
                self.idle_machine.succeed()
                self.idle_machine = self.env.event()
            # print(self.env.now, '  Wait for checking')
            yield self.wait_before_check  # wait for time to check(valve)
            # machine 이 반납된 후마다 열렸다가 바로 닫힘

    def check_part_exist(self):
        while True:
            if len(self.buffer_to_machine.items) != 0:
                self.parts_exist.succeed()
                self.parts_exist = self.env.event()
            yield self.wait_before_new_arrival


    def _to_process(self):
        while True:
            part = yield self.buffer_to_process.get()
            # print(str(self.env.now) + '  ' + part.id + '  Now ready for next process')

            # next process
            next_process_name = self.process_list[part.step + 1]
            # print(str(self.env.now) + '  ' + part.id + '  next process is : '+ next_process_name)
            next_process = self.model[next_process_name]

            if next_process.__class__.__name__ == 'Process':
                # part transfer
                next_process.buffer_to_machine.put(part)
                # print(str(self.env.now) + '  ' + part.id + '  part_transferred to  ' + next_process.name)
            else:
                part.completion_time = self.env.now
                next_process.put(part)

            part.step += 1
            self.parts_sent += 1

    # i : idle machine index , j : part in buffer_to_machine index
    # idle_machines (list) 와 parts_in_buffer (list) 이용해서 indexing
    # idle machine name 과 part in buffer_to_machine id
    def routing(self, action):
        if action == 0:  # WSPT
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[i.idx]
                        w_ij = p_ij / j.weight
                        W_i.append(w_ij)
                    i.W_ij = min(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[i.idx]
                        w_ij = p_ij / j.weight
                        W_j.append(w_ij)
                    j.W_ij = min(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij)

            elif len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) == 0:
                self.machine_store.items.sort(key=lambda machine: machine.priority)


        elif action == 1:  # WMDD
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[i.idx]
                        w_ij = max(p_ij, j.due_date - self.env.now)
                        w_ij = w_ij / j.weight
                        W_i.append(w_ij)
                    i.W_ij = min(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[i.idx]
                        w_ij = max(p_ij, j.due_date - self.env.now)
                        w_ij = w_ij / j.weight
                        W_j.append(w_ij)
                    j.W_ij = min(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij)

            elif len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) == 0:
                self.machine_store.items.sort(key=lambda machine: machine.priority)

            return 1

        elif action == 2:  # ATC
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                p = 0.0  # average nominal processing time
                for part in self.buffer_to_machine.items:
                    p += part.process_time
                p = p / len(self.buffer_to_machine.items)

                h = 2  # look-ahead parameter

                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[i.idx]
                        w_ij = -1 * max(0, j.due_date - self.env.now - p_ij) / (h * p)
                        w_ij = j.weight / p_ij * math.exp(w_ij)
                        W_i.append(w_ij)
                    i.W_ij = max(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij, reverse=True)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[i.idx]
                        w_ij = -1 * max(0, j.due_date - self.env.now - p_ij) / (h * p)
                        w_ij = j.weight / p_ij * math.exp(w_ij)
                        W_j.append(w_ij)
                    j.W_ij = max(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij, reverse=True)

            elif len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) == 0:
                self.machine_store.items.sort(key=lambda machine: machine.priority)


        elif action == 3:  # WCOVERT
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                K_t = 2  # approximation factor

                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[i.idx]
                        w_ij = 1 - max(0, j.due_date - self.env.now - p_ij) / (K_t * p_ij)
                        w_ij = j.weight / p_ij * max(0, w_ij)
                        W_i.append(w_ij)
                    i.W_ij = max(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij, reverse=True)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[i.idx]
                        w_ij = 1 - max(0, j.due_date - self.env.now - p_ij) / (K_t * p_ij)
                        w_ij = j.weight / p_ij * max(0, w_ij)
                        W_j.append(w_ij)
                    j.W_ij = max(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij, reverse=True)

            elif len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) == 0:
                self.machine_store.items.sort(key=lambda machine: machine.priority)


class Machine(object):
    def __init__(self, env, process_name, name, idx, priority, out, model):
        self.env = env
        self.process_name = process_name
        self.name = name
        self.idx = idx
        self.priority = priority
        self.out = out
        self.model = model

        self.machine = simpy.Store(env, capacity=1)

        self.start_work = None
        self.working_start = 0.0

        self.W_ij = None

    def work(self, part, machine_store):
        # process_time
        proc_time = part.p_ij[self.idx]
        proc_time = np.random.exponential(proc_time)
        part.real_proc_time = proc_time

        self.machine.put(part)
        self.start_work = self.env.now
        self.working_start = self.env.now
        yield self.env.timeout(proc_time)

        # get parts from machine after process has finished
        part = yield self.machine.get()

        # put part to the buffer_to_process(out buffer)
        # transfer to '_to_process' function
        self.out.put(part)

        # return machine object(self) to the machine store
        machine_store.put(self)

        # Idle machine occurs -> check
        # Idle machine check 시점이 되면 wait_before_check(valve)를 열었다가 바로 닫는다
        self.model[self.process_name].wait_before_check.succeed()
        self.model[self.process_name].wait_before_check = self.env.event()
        ################################################################################################################


class Sink(object):
    def __init__(self, env):
        self.env = env
        self.name = 'Sink'

        # self.tp_store = simpy.FilterStore(env)  # transporter가 입고 - 출고 될 store
        self.parts_rec = 0
        self.last_arrival = 0.0
        self.sink = list()

    def put(self, part):
        self.parts_rec += 1
        self.last_arrival = self.env.now
        self.sink.append(part)
        # print(str(self.env.now) + '  ' + self.name + '  ' + part.id + '  completed')
