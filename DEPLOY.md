# 部署说明（Streamlit Community Cloud 等）

## 只上传 `.py` 和两个表格够不够？

不够「自动跑起来」。你需要：

1. **代码托管**：例如 GitHub 仓库里有本项目的 Python 源码、`requirements.txt`、`questions.xlsx`、`scaffolding.xlsx`。
2. **托管运行环境**：用 **Streamlit Community Cloud**（或其它支持 Streamlit 的平台）从仓库 **部署应用**，在平台里配置 **Secrets**，应用才会在公网地址上运行并调用 DeepSeek。

仅把文件拷到网盘或只建空仓库而不部署，别人无法通过链接使用 Tutor。

## 仓库建议包含的文件

| 文件 | 说明 |
|------|------|
| `main.py` 及所有 `.py` 模块 | 应用入口与逻辑 |
| `requirements.txt` | 云端安装依赖 |
| `questions.xlsx` | 题目（必需） |
| `scaffolding.xlsx` | 脚手架（必需） |
| `.streamlit/secrets.toml` | **不要提交**；密钥只在平台「Secrets」里配置 |

不要把含真实 Key 的 `secrets.toml` 推送到公开仓库。

## Streamlit Cloud：Secrets（TOML）

在应用的 **Settings → Secrets**（或 Advanced settings → Secrets）中填写 **TOML**，键名须与下面代码读取的名字 **完全一致**（区分大小写）：

```toml
DEEPSEEK_API_KEY = "sk-你的DeepSeek密钥"
```

保存后等待约一分钟再刷新应用。进入应用后选择 **「使用平台提供的连接」**（hosted），即会使用该 Key。

也可二选一备用名（本程序会按顺序尝试）：

```toml
# 任选其一即可，不必三个都写
DEEPSEEK_API_KEY = "sk-..."
# OPENAI_API_KEY = "sk-..."
```

### 常见错误：把 Key 写在「段落」里

下面这种 **TOML 表** 会把 `DEEPSEEK_API_KEY` 放在**二级**，旧版 Streamlit 不会把它同步到环境变量 `DEEPSEEK_API_KEY`，容易导致应用判定「未配置托管 Key」。请优先改成**顶层一行**（见上一节）。

```toml
# 不推荐（易读不到）；若你已这样写，新版本 tutor_api_keys 会尝试从子表里查找
[secrets]
DEEPSEEK_API_KEY = "sk-..."
```

正确示例（**无** `[xxx]` 包裹）：

```toml
DEEPSEEK_API_KEY = "sk-..."
```

### Cloud 上要和「当前应用」绑定

Secrets 填在 **这个应用** 的 Settings 里（不是别的仓库、也不是只改本地文件未 push）。改完后等约 1 分钟并让 Cloud **重新部署** 一次更稳妥。

## 最小示例：从 Secrets 调 DeepSeek（OpenAI 兼容）

下面是一段 **独立小例子**，用于理解「平台 Secrets → 客户端」的关系。**本仓库正式逻辑**在 `tutor_api_keys.py` 的 `get_built_in_api_key()` 与 `llm_api.py` 的 `configure_llm()` 中，已按同样方式读取 `st.secrets["DEEPSEEK_API_KEY"]` 等，无需在 `main.py` 里再写一遍。

```python
import streamlit as st
import openai  # DeepSeek 兼容 OpenAI 接口

# 从 Secrets 中读取 API Key（键名必须与 Cloud 里 TOML 完全一致）
client = openai.OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "用一句话说你好。"}],
)
st.write(resp.choices[0].message.content)
```

部署时把上述保存为任意 `pages/xxx.py` 或单独小应用即可做连通性测试；本数学 Tutor 仍以 `main.py` 为入口。

## 常见问题

- **仍提示「未配置平台托管 Key」**  
  检查：① `DEEPSEEK_API_KEY` 是否在 **TOML 最外层**（不要包在 `[secrets]` 等表下面，除非已更新到最新 `tutor_api_keys.py`，其会尝试读取子表）；② 键名大小写、英文双引号；③ Secrets 是否保存在 **当前 Cloud 应用** 的设置里；④ 保存后等待约 1 分钟并 **Redeploy**；⑤ GitHub 上是否已包含最新的 `tutor_api_keys.py`。

- **费用**  
  使用平台托管 Key 时，调用 DeepSeek 的费用计入 **你** 在 DeepSeek 控制台对应账户。
