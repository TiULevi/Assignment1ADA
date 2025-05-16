from flask import Flask, request, jsonify

app = Flask(__name__)
PATIENTS = {}

@app.route('/patients', methods=['POST'])
def add_patient():
    data = request.get_json()
    pid = data.get('id')
    PATIENTS[pid] = data
    return jsonify({"status": "created", "patient": data}), 201

@app.route('/patients/<pid>', methods=['GET'])
def get_patient(pid):
    patient = PATIENTS.get(pid)
    if not patient:
        return jsonify({"error": "Not found"}), 404
    return jsonify(patient), 200

@app.route('/patients/<pid>', methods=['PUT'])
def update_patient(pid):
    data = request.get_json()
    PATIENTS[pid] = data
    return jsonify({"status": "updated", "patient": data}), 200

@app.route('/patients/<pid>', methods=['DELETE'])
def delete_patient(pid):
    if pid in PATIENTS:
        del PATIENTS[pid]
        return jsonify({"status": "deleted"}), 200
    return jsonify({"error": "Not found"}), 404

@app.route('/patients/<pid>/history', methods=['GET'])
def get_history(pid):
    # Simulate patient history
    return jsonify({"history": ["scan1", "scan2"]}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
