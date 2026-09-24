#!/bin/bash
source venv/bin/activate

echo "Starting Server..."
python server.py &
sleep 3

echo "Starting Hospital A Client..."
python client.py --hospital-id A &

echo "Starting Hospital B Client..."
python client.py --hospital-id B &

echo "Starting Hospital C Client..."
python client.py --hospital-id C &

# Wait for all background processes to finish
wait
