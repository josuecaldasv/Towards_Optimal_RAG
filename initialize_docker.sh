#!/bin/bash

CONTAINER_NAME="optimal_rag"
IMAGE_NAME="user/optimal_rag"
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
docker rmi $IMAGE_NAME
cd Towards_Optimal_RAG
docker build -t $IMAGE_NAME .
docker run -d --name $CONTAINER_NAME -p 10002:10002 --ipc=host $IMAGE_NAME
echo "Container $CONTAINER_NAME is now up and running on port 10002!"