#!/bin/bash

localBranch=''
remoteBranch=''

if [ $# -eq 1 ]; then
	localBranch=$1
	remoteBranch='master'
elif [ $# -eq 2 ]; then
	localBranch=$2
	remoteBranch=$1
else
	echo "Usage:"
	echo " $0 <local branch> <remote branch>"
	echo " $0 <local branch>"
	exit 1
fi


git checkout $remoteBranch
[ $? -ne 0 ] && exit 1
git pull
[ $? -ne 0 ] && exit 1
git checkout $localBranch
[ $? -ne 0 ] && exit 1

git stash
[ $? -ne 0 ] && exit 1
git rebase $remoteBranch
[ $? -ne 0 ] && exit 1
git push origin $localBranch:$remoteBranch
[ $? -ne 0 ] && exit 1

git stash apply --index
[ $? -eq 0 ] && git stash drop

git checkout $remoteBranch
[ $? -ne 0 ] && exit 1
git pull
[ $? -ne 0 ] && exit 1
git checkout $localBranch
[ $? -ne 0 ] && exit 1

echo "OK!"

