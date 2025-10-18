#!/bin/bash

# NFL Prediction Model - GitHub Push Helper
# This script helps you push your code to GitHub

echo "=============================================="
echo "🏈 NFL Prediction Model - GitHub Push"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "PRD.md" ]; then
    echo "❌ Error: Run this script from the project root directory"
    exit 1
fi

# Check if GitHub username is provided
if [ -z "$1" ]; then
    echo "Usage: ./push-to-github.sh YOUR_GITHUB_USERNAME"
    echo ""
    echo "Example: ./push-to-github.sh joemartineziv"
    echo ""
    echo "Or manually run these commands:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/nfl-prediction-model.git"
    echo "  git push -u origin main"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_URL="https://github.com/$GITHUB_USERNAME/nfl-prediction-model.git"

echo "GitHub Username: $GITHUB_USERNAME"
echo "Repository URL: $REPO_URL"
echo ""
echo "⚠️  Make sure you've created the repository on GitHub first!"
echo "    Go to: https://github.com/new"
echo "    Repository name: nfl-prediction-model"
echo "    Do NOT initialize with README, .gitignore, or license"
echo ""
read -p "Have you created the repository? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please create the repository first, then run this script again."
    exit 1
fi

echo ""
echo "Adding remote repository..."
git remote add origin "$REPO_URL" 2>/dev/null || {
    echo "Remote already exists. Updating..."
    git remote set-url origin "$REPO_URL"
}

echo ""
echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ Successfully pushed to GitHub!"
    echo "=============================================="
    echo ""
    echo "Your repository is now available at:"
    echo "https://github.com/$GITHUB_USERNAME/nfl-prediction-model"
    echo ""
    echo "Next steps:"
    echo "  1. View your repo: https://github.com/$GITHUB_USERNAME/nfl-prediction-model"
    echo "  2. Add topics: machine-learning, nfl, python, predictions"
    echo "  3. Create a nice README badge"
    echo "  4. Share with the world! 🎉"
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "  1. Repository doesn't exist on GitHub"
    echo "  2. Authentication failed (check your GitHub credentials)"
    echo "  3. Remote URL is incorrect"
    echo ""
    echo "Try these commands manually:"
    echo "  git remote -v  # Check remote URL"
    echo "  git push -u origin main  # Try pushing again"
fi

