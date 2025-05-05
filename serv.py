#!/usr/bin/env python3
import os
import sys
import json
import base64
import requests

def send_file(filename, run_id, url="http://192.168.1.11:5001/validate"):
    # Read and encode the file
    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "file": encoded,
        "run_id": run_id
    }
    headers = {"Content-Type": "application/json"}
    
    # Send the request
    resp = requests.post(url, json=payload, headers=headers)
    
    # Print out status and response
    print(f"Status: {resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <filename> <run_id>")
        sys.exit(1)
    filename = sys.argv[1]
    run_id = sys.argv[2]
    if not os.path.isfile(filename):
        print(f"Error: file '{filename}' does not exist.")
        sys.exit(1)
    send_file(filename, run_id)
