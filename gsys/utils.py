# Copyright 2023 Yuhao Zhang and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from .constants import TIME_FORMAT
# import sys
import datetime
DEBUG = True


def tstamp():
    return datetime.datetime.now().strftime(TIME_FORMAT)


def hfill(sentence='', char='-', limit=79):
    start = (limit - len(sentence)) // 2
    end = start + len(sentence)
    first_part = "".join(['-' for _ in range(start)])
    second_part = "".join(['-' for _ in range(end, limit)])
    print("{}{}{}".format(first_part, sentence, second_part))


def logs(message):
    timestamp = tstamp()
    printed = "{}: {}".format(message, timestamp)
    print(printed)
    # sys.stdout.flush()
    return printed


def log_local_test_acc(*args, submodel=0):
    template = \
        "EVENT<Machine: {}, Epoch: {}, Mode: {}, Accuracy: {}, Count: {}, Submodel: {}>"
    message = template.format(*args, submodel)
    logs(message)
    return message


def DiskLogs(filenames):
    def logs_disk(message):
        printed = logs(message)
        for filename in filenames:
            with open(filename, 'a') as f:
                f.write(printed + '\n')
    return logs_disk


def timeit_factory(debug=DEBUG):
    def timeit(func):
        def timed(*args, **kwargs):
            function_name = func.__name__
            if debug:
                logs("Start inside {}".format(function_name))
            result = func(*args, **kwargs)
            if debug:
                logs("End inside {}".format(function_name))
            return result
        return timed
    return timeit


class logsc(object):
    def __init__(self,
                 log,
                 debug=DEBUG,
                 logs_fn=logs,
                 elapsed_time=False,
                 log_dict={},
                 accumulate=False):
        self.log = log
        self.debug = debug
        self.logs_fn = logs_fn
        self.elapsed_time = elapsed_time
        self.log_dict = log_dict
        self.accumulate = accumulate

    def __enter__(self):
        self.start = datetime.datetime.now()
        if self.debug:
            self.logs_fn("Start {}".format(self.log))

    def __exit__(self, type, value, traceback):
        self.end = datetime.datetime.now()
        if self.debug:
            self.logs_fn("End {}".format(self.log))
        if self.elapsed_time:
            elapsed = (self.end - self.start).total_seconds()
            print(
                "ELAPSED TIME: {}".format(
                    elapsed
                )
            )
            if self.accumulate and self.log in self.log_dict:
                self.log_dict[self.log] += elapsed
            else:
                self.log_dict[self.log] = elapsed


class hfillc(object):
    def __init__(self,
                 sentence):
        self.sentence = sentence

    def __enter__(self):
        hfill(self.sentence)

    def __exit__(self, type, value, traceback):
        hfill()
