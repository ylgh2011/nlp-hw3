#!/bin/bash

remoteBranch=''
localBranch=''

if [ $# -eq 1 ]; then
	remoteBranch='master'
	localBranch=$1
elif [ $# -eq 2 ]; then
	remoteBranch=$1
	localBranch=$2
else
	echo "Usage:"
	echo " $0 <remote branch> <local branch>"
	echo " $0 <local branch>"
	exit 1
fi


git checkout $localBranch
[ $? -ne 0 ] && exit 1
git stash
[ $? -ne 0 ] && exit 1

git checkout $remoteBranch
[ $? -ne 0 ] && exit 1
git pull
[ $? -ne 0 ] && exit 1
git checkout $localBranch
[ $? -ne 0 ] && exit 1

git rebase $remoteBranch
[ $? -ne 0 ] && exit 1

git stash apply --index
[ $? -eq 0 ] && git stash drop

if [ $? -ne 0 ]; then
	echo "Please check stash status"
else
	echo "OK!"
fi
