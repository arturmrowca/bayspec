import pandas as pd
import copy
import re


def single_BN_log(tscbn, traces_per_log, samples_per_trace):
    if traces_per_log < 1 or samples_per_trace < 1:
        print("Parameters must be greater than zero!")
        return None

    total_samples = traces_per_log * samples_per_trace

    trace_list = _one_BN_trace_list(tscbn, total_samples, samples_per_trace)

    true_traces = len(trace_list)

    true_traces_per_log = true_traces // samples_per_trace

    log = []
    for i in range(true_traces_per_log):
        trace = pd.concat(trace_list[i*samples_per_trace: (i+1)*samples_per_trace])
        trace = trace["value"]
        log.append(copy.deepcopy(trace))

    return log


def multi_BN_log_interleaving(tscbn_list, traces_per_log, samples_per_trace):
    if traces_per_log < 1 or samples_per_trace < 1:
        print("Parameters must be greater than zero!")
        return None

    samples_per_bn = traces_per_log * samples_per_trace

    traces = [[] for _ in range(traces_per_log)]
    for i, tscbn in enumerate(tscbn_list):
        trace_list = _one_BN_trace_list(tscbn, samples_per_bn, samples_per_trace, i)

        true_traces = len(trace_list)
        true_traces_per_log = true_traces // samples_per_trace

        for j in range(true_traces_per_log):
            traces[j].extend(trace_list[j*samples_per_trace: (j+1)*samples_per_trace])

    log = []
    for i, trace in enumerate(traces):
        trace_df = pd.concat(trace)
        trace_df = trace_df.sort_values("timestamp")
        trace_df = trace_df["value"]
        log.append(copy.deepcopy(trace_df))

    return log


def _one_BN_trace_list(tscbn, total_samples, samples_per_trace, bn_number=0):
    # Sampling from Model and generate Traces
    evidence = {}  # evidence when sampling
    in_seq = tscbn.randomsample(total_samples, evidence)
    trace_list = _sequences_to_traces(in_seq, tscbn.Vdata, bn_number, samples_per_trace)

    return trace_list


def _sequences_to_traces(random_samples, vertex_data, bn_number, samples_per_trace, sep="_"):
    sequences = _sequences_to_intervals_keep_same(random_samples, vertex_data, sep)
    trace_list = []
    max_timestamp = 0
    for i, sequence in enumerate(sequences[0]):
        if i % samples_per_trace == 0:
            max_timestamp = 0
        dfs = []
        for node in sequence:
            data = [["F{}.{}={}".format(bn_number, node, s[0]), s[1] + max_timestamp] for s in sequence[node]]
            dfs.append(pd.DataFrame(data, columns=["value", "timestamp"]))
        trace = pd.concat(dfs)
        trace = trace.sort_values("timestamp")
        max_timestamp = trace["timestamp"].max() + 0.1
        trace_list += [trace]

    return trace_list


def _sequences_to_intervals_keep_same(random_samples, vertex_data, sep="_"):
    iq = 0
    outputs = []
    in_seq = []
    for s in random_samples:
        invalid = False
        iq += 1
        x_start, x_stop, y, label, map = [], [], [], [], dict()

        # get abs. parent_times for interval start
        abs_time = {}
        k_to_proc = list(s.keys())
        i = -1

        while True:
            if len(k_to_proc) == 0: break
            i += 1
            if i >= len(k_to_proc): i = 0
            k = k_to_proc[i]
            if str.startswith(k, "dL_"):
                k_to_proc.remove(k)
                continue

                # parent time is max of parents time  + my tprev
            try:
                # look for value for abs_time of all parents
                rel_time_to_me = s["dL_" + k]

                if not vertex_data[k]["parents"]:
                    pars = []
                else:
                    pars = vertex_data[k]["parents"]

                found_time = [0]
                for p in pars:
                    found_time.append(abs_time[p])

                abs_time_of_my_earliest_parent = max(found_time)  # [0]+[abs_time[p[3:]] for p in pars])

                abs_time[k] = rel_time_to_me + abs_time_of_my_earliest_parent
                k_to_proc.remove(k)
            except:
                pass  # print("Stuck") # not all parents there yet

        # plot this sample - abs_time[k] is absolute time of tprev (mean)
        overall_high = max(list(abs_time.values())) + 0.5 * max(list(abs_time.values()))  # buffer indicating end
        output_intervals = {} # key: TV value: list of tuples (value, start, end)
        for el in s:
            if str.startswith(el, "dL_"): continue

            if not sep:
                span = re.search("\d", el)
                name = el[:span.start()]
                number = int(el[span.start():])
            else:
                name = el.rsplit(sep,1)[0]
                number = int(el.rsplit(sep,1)[1])

            x_s = abs_time[el]
            try:
                if sep:
                    x_e = abs_time[name + sep + str(number + 1)]  #
                else:
                    x_e = abs_time[name + str(number + 1)]  #
            except:
                x_e = overall_high  # no next element

            #yy = name
            lab = s[el]
            yy =lab
            if x_e < 0:
                invalid = True
                break
            if x_s > x_e:
                a = 0
                invalid = True
                break

            x_start.append(x_s)
            x_stop.append(x_e)
            y.append(name)
            label.append(lab)

            try:
                output_intervals[name].append([yy, x_s, x_e])
            except:
                output_intervals[name] = [[yy, x_s, x_e]]
        if not invalid:
            in_seq.append(s)
            # remove nevers
            for n in output_intervals:
                occs = output_intervals[n]
                t_rem = []
                i=0
                for _ in range(len(occs)):
                    o = occs[i]

                    # if previous element same as mine I would not see it
                    pre_el_idx = i - 1
                    if pre_el_idx < 0:
                        i += 1
                        continue

                    i += 1
            outputs.append(output_intervals)

            # drop nevers
            i = 0
            while "Never" in label:
                idx = label.index("Never")
                # replace with previous - defined by my endinterval
                # lösche meinen Start - finde zugehörigen end
                # x_start 5   10 20
                # x_stop  10  20 40
                # val     O   N  O

                repl_idx = x_stop.index(x_start[idx])
                x_stop[repl_idx] = x_stop[idx]
                del x_start[idx]
                del x_stop[idx]
                del y[idx]
                del label[idx]

    return outputs, in_seq
