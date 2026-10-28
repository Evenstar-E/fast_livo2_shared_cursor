# 上传 GitHub

上传根目录应为本精简目录，不是原来包含数据和历史实验的工作区。这里没有 .git 历史或远程配置，不会误推到上游 hku-mars。

先在 GitHub 创建你自己的空仓库，再在本目录运行以下命令（替换占位 URL）：

```bash
git init -b main
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare source-only recovery research snapshot"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

本次整理没有执行 commit、创建远程仓库或 push。不要复制原项目 .git；它的 origin 指向上游。不要使用 `git add -f` 添加被忽略的数据/权重/私有配置。

目前适合作为研究开发源代码分享，不宜称为完整论文复现发布。完整 ROS 构建、冻结模型权重和隔离实验复现情况以 VALIDATION/REPRODUCIBILITY 为准。
