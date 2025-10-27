# 1. Generate a new SSH key (press Enter for defaults)
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. Start the SSH agent
eval "$(ssh-agent -s)"

# 3. Add your key to the agent
ssh-add ~/.ssh/id_ed25519

# 4. Copy your SSH public key to clipboard
cat ~/.ssh/id_ed25519.pub
