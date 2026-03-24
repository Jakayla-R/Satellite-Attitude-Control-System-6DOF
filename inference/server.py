from flask import Flask, request, jsonify
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.mingpt_6dof import DecisionTransformer6DOF

app = Flask(__name__)

K = 20
model = DecisionTransformer6DOF(K=K)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), '..', 'model', 'checkpoints', 'dt_6dof.pt'),
    map_location='cpu'
))
model.eval()

@app.route('/action', methods=['POST'])
def get_action():
    data = request.json
    states  = np.array(data['states'],  dtype=np.float32)   # (K, 13)
    actions = np.array(data['actions'], dtype=np.float32)   # (K, 6)
    rtgs    = np.array(data['rtgs'],    dtype=np.float32)   # (K, 1)
    action, uncertainty = model.get_action(states, actions, rtgs)
    return jsonify({
        'action':      action.tolist(),
        'uncertainty': uncertainty.tolist()
    })

if __name__ == '__main__':
    app.run(port=5050)