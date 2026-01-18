#!/bin/bash
# Test script to verify GPU usage during inference

# Cleanup function to kill background processes
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null
    exit 0
}

# Trap SIGINT (Ctrl-C) and SIGTERM to call cleanup
trap cleanup SIGINT SIGTERM

echo "Starting GPU monitoring..."
echo "Make sure the server is running on port 8080"
echo "Press Ctrl-C to stop"

# Monitor GPU in background
(
    while true; do
        clear
        nvidia-smi
        sleep 1
    done
) &
MONITOR_PID=$!

# Wait a moment
sleep 2

echo ""
echo "Making inference requests..."
echo ""

# Make several requests
for i in {1..20}; do
    echo "Request $i..."
    curl -s -X POST http://localhost:8080/txt/embed \
        -H 'content-type: application/json' \
        -d '{"input": "This is a test request to trigger GPU inference '$(date +%s)'"}' > /dev/null
    sleep 0.5
done

echo ""
echo "Making batch requests..."
for i in {1..5}; do
    echo "Batch request $i..."
    curl -s -X POST http://localhost:8080/txt/batch \
        -H 'content-type: application/json' \
        -d '{"inputs": ["test 1", "test 2", "test 3", "test 4", "test 5"]}' > /dev/null
    sleep 0.5
done

echo ""
echo "Making image requests..."
for i in {1..5}; do
    echo "Image request $i..."
    curl -s -X POST http://localhost:8080/img/embed \
        -H 'content-type: application/json' \
        -d '{"content": "https://picsum.photos/400/300"}' > /dev/null
    sleep 0.5
done

# Stop monitoring
cleanup

echo ""
echo "Test complete. Check the nvidia-smi output above for GPU usage."

