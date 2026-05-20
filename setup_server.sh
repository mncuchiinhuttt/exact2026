#!/bin/bash
set -e

echo "Updating package list..."
sudo apt-get update

echo "Installing tmux, python3, python3-pip, python3-venv, and git..."
sudo apt-get install -y tmux python3 python3-pip python3-venv git

echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements.txt..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi

echo "Installing requirements-server.txt..."
if [ -f requirements-server.txt ]; then
    pip install -r requirements-server.txt
else
    echo "requirements-server.txt not found!"
fi

echo "Installing vLLM..."
pip install vllm

echo "Configuring Git..."
read -p "Enter Git User Name (or press enter to skip): " GIT_NAME
if [ ! -z "$GIT_NAME" ]; then
    git config --global user.name "$GIT_NAME"
fi

read -p "Enter Git User Email (or press enter to skip): " GIT_EMAIL
if [ ! -z "$GIT_EMAIL" ]; then
    git config --global user.email "$GIT_EMAIL"
fi

echo "Installing Cloudflare Tunnel (cloudflared)..."
if ! command -v cloudflared &> /dev/null; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
else
    echo "cloudflared is already installed."
fi

echo "Setting up Cloudflare Tunnel..."
read -p "Do you want to log in to Cloudflare Tunnel now? (y/n): " CF_LOGIN
if [ "$CF_LOGIN" = "y" ] || [ "$CF_LOGIN" = "Y" ]; then
    cloudflared tunnel login
fi

echo "Setup complete!"
echo "Remember to activate the virtual environment with 'source venv/bin/activate' before running your Python scripts."
