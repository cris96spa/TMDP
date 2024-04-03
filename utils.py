import json
import numpy as np
import os

res_path = './results'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def to_json(dict, indent):
    return json.dumps(dict, indent = indent, cls=NpEncoder)

def from_json(json_object):
    return json.loads(json_object)

def write_results(size, nA, env_name, results, mode):
    file_name = env_name + "_" + str(size) + "_" + str(nA)
    file_name = os.path.join(res_path, file_name)
    with open(file_name, mode = mode) as file:
        file.write(json.dumps(results, indent = 4, cls=NpEncoder))

def read_results(size, nA, env_name):
    file_name = env_name + "_" + str(size) + "_" + str(nA)
    file_name = os.path.join(res_path, file_name)
    results = {}
    with open(file_name, 'r') as file:
        results = json.load(file)
    parse_results(results)
    return results

def parse_results(results):
    for res in results:
        for key in res.keys():
            """ if isinstance(res[key], int):
                res[key] = np.integer(res[key])
            if isinstance(res[key], float):
                res[key] = np.floating(res[key]) """
            if isinstance(res[key], list):
                res[key] = np.asarray(res[key])

def aggregate_results(results, different_count, z=1.96):
    res = []
    for i, r in enumerate(results):
        if i < different_count:
            res.append(r)
            res[i]['J'] = [res[i]['J']]
            res[i]['delta_q'] = [res[i]['delta_q']]
        else:
            res[i % different_count]['J'].append(r['J'])
            res[i % different_count]['delta_q'].append(r['delta_q'])

        for r in res:
            r['avg_J'] = np.mean(r['J'])
            r['std_J'] = np.std(r['J'])
            r['ci_J'] = z * r['std_J']/np.sqrt(len(r['J']))
            r['avg_delta_q'] = np.mean(r['delta_q'])
            r['std_delta_q'] = np.std(r['delta_q'])
            r['ci_delta_q'] = z * r['std_delta_q']/np.sqrt(len(r['J']))
    return res