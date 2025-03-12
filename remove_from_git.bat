@echo off
chcp 65001
echo 正在从Git索引中移除指定文件和文件夹...

git rm --cached train_debug.py
git rm --cached test_imports.py
git rm --cached test_evaluate.py
git rm --cached create_project_structure.bat
git rm --cached train_single_env.py
git rm --cached -r debug_output
git rm --cached -r output
git rm --cached -r quick_output
git rm --cached -r test_output
git rm --cached -r .idea

echo 正在将文件和文件夹添加到.gitignore...

echo train_debug.py >> .gitignore
echo test_imports.py >> .gitignore
echo test_evaluate.py >> .gitignore
echo create_project_structure.bat >> .gitignore
echo train_single_env.py >> .gitignore
echo debug_output/ >> .gitignore
echo output/ >> .gitignore
echo quick_output/ >> .gitignore
echo test_output/ >> .gitignore
echo .idea/ >> .gitignore

echo 正在提交更改...

git commit -m "从仓库中移除指定文件和文件夹，但保留本地副本"

echo 正在添加.gitignore文件...

git add .gitignore
git commit -m "将指定文件和文件夹添加到.gitignore"

FOR /F "tokens=*" %%g IN ('git branch --show-current') do (SET current_branch=%%g)
echo 当前本地分支是: %current_branch%

set /p local_branch=请输入本地分支名称(默认为当前分支 %current_branch%):
if "%local_branch%"=="" set local_branch=%current_branch%

set /p remote_branch=请输入远程分支名称(默认与本地分支相同 %local_branch%):
if "%remote_branch%"=="" set remote_branch=%local_branch%

echo 正在将本地分支 %local_branch% 推送到远程分支 %remote_branch%...
git push origin %local_branch%:%remote_branch%

echo 完成！
pause