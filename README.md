# web2api

一个本地 API 网关：把 DeepSeek 网页版对话能力，转为标准 **OpenAI 兼容接口**供本地调用。

原理：各家 AI 平台（DeepSeek 等）的网页聊天通常只面向浏览器场景，其私有接口与 OpenAI 标准格式不兼容。本工具在中间做一层协议转换——用你网页版的登录态（token）调用其私有接口，再把返回**翻译成 OpenAI 标准格式**。于是任何支持"填 API 地址 + key"的客户端（Cherry Studio、各类 Agent、IDE 插件）都能直接连上使用。

## 免责声明

本项目**仅供个人学习与技术研究**，用于理解逆向协议与 API 网关实现。使用前请知悉：

- 本工具会以你的账号登录态调用平台接口，可能触发平台的风控策略（限流、账号限制等），**后果需自行承担**。
- 请自行阅读并遵守对应平台的服务条款（Terms of Service）。**作者不对任何使用行为及后果承担责任**。
- token 等同于账号登录态，请勿外泄或随项目分发。
- 禁止用于任何形式的商业转售或批量服务提供。

## 技术栈

Python + FastAPI，无 GUI、无账号池。当前实现了 DeepSeek 适配器（调用链清晰，适合作为新平台适配的范本）。

**依赖**：Python 3.10+、`pip install fastapi uvicorn httpx pyyaml wasmtime`。PoW 求解用 wasmtime 直接跑 DeepSeek 官方 WASM——算法是私有的 23 轮 Keccak，纯 Python 重写慢且易错，直接跑浏览器同款 WASM 保证一致，**无需 Node.js**。

## 文件

| 文件 | 作用 |
|------|------|
| `server.py` | 入口。FastAPI 服务器，暴露 `/v1/chat/completions` `/v1/models` `/health` |
| `deepseek.py` | DeepSeek 适配器。把 OpenAI 请求翻译成 DeepSeek 私有接口调用（换token→建会话→PoW→发对话→删会话） |
| `sse.py` | 解析 DeepSeek 私有 SSE，抽出正文，转成 OpenAI chunk |
| `pow.py` | DeepSeekHashV1 工作量证明求解（wasmtime 驱动官方 WASM，无 Node 依赖） |
| `providers/deepseek_pow.wasm` | DeepSeek 官方 PoW WASM（浏览器同款，保证逐字节一致） |
| `throttle.py` | 上游限流（串行 + 最小间隔），防触发风控 |
| `config.yaml` | 配置：token、监听地址、端口、访问密钥 |

## 用法

### 源码运行
1. **拿 token**：登录 https://chat.deepseek.com → F12 → Application → Local Storage → `userToken` → 复制值。
2. **填配置**：把 token 粘进 `config.yaml` 的 `deepseek.token`。
3. **启动**：
   ```bash
   pip install fastapi uvicorn httpx pyyaml wasmtime
   python server.py
   ```
4. **接入**：任意 OpenAI 客户端，把 base URL 指向 `http://127.0.0.1:8080/v1`，key 随便填（除非你在 config 里设了 `api_key`）。

### exe 使用指南（Windows 单文件版，`dist/web2api.exe`）

一个文件双击即用，**不需要安装 Python、Node 或任何依赖**。首次拿到后按下面四步走：

**① 双击运行一次**
- 双击 `web2api.exe`，弹出黑色命令行窗口，闪一下后自动关闭。
- 这**不是崩溃**——它在 exe 旁边生成了 `config.yaml` 配置文件，然后退出等你填 token。

**② 获取 DeepSeek token**
1. 浏览器打开 https://chat.deepseek.com 并登录你的账号。
2. 按 `F12` 打开开发者工具。
3. 切到 **Application**（应用）标签 → 左侧展开 **Local Storage** → 点 `https://chat.deepseek.com`。
4. 在右侧列表找到 `userToken`，双击它的 Value 全选复制（一长串字符）。

**③ 把 token 填进配置**
1. 右键 `config.yaml` → 打开方式 → 记事本。
2. 找到这一行：
   ```yaml
   token: "PASTE_YOUR_DEEPSEEK_USER_TOKEN_HERE"
   ```
3. 把引号里的占位符**整段**替换成刚复制的 token，保存关闭。

**④ 再次双击运行**
- 黑色窗口会**保持打开**，显示类似：
  ```
  web2api listening on http://127.0.0.1:8080
  ```
  这就成功了。窗口别关（关了服务就停），最小化即可。
- 想停止服务：直接关掉这个窗口。

**接入客户端（以 Cherry Studio 为例）**
| 配置项 | 填什么 |
|---|---|
| API 地址（Base URL） | `http://127.0.0.1:8080/v1` |
| API Key | 随便填一个字符串（默认不校验；只有改了 `api_key` 才需要填对） |
| 模型 | `DeepSeek-V4.1-Flash` 或 `deepseek-v4-pro`（在 config.yaml 的 `models` 里配置） |

其他任何支持"自定义 OpenAI 接口"的工具（各类 Agent、IDE 插件、命令行工具）填法相同。

**手动验证（可选，命令行）**
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

**常见问题**
| 现象 | 原因 / 解法 |
|---|---|
| 双击后窗口一闪就没，也没生成 config.yaml | 被 Windows SmartScreen 拦截。点"更多信息"→"仍要运行"；或右键 exe → 属性 → 底部勾"解除锁定"→ 确定 |
| 对话返回 `invalid token` | token 失效（DeepSeek 网页端重新登录会刷新）。回到第②步重新复制 |
| 启动后窗口一闪退出，提示端口占用 | 8080 被别的程序占了。改 config.yaml 里的 `port`（如 `8081`），客户端地址也要跟着改 |
| 改了 config.yaml 不生效 | 必须**重启 exe** 才会重读配置 |

**安全提醒**
- `config.yaml` 里的 token **等同于你的 DeepSeek 账号登录态**——只发 exe 给别人，别把你这份 config.yaml 一起发出去。
- 默认只监听本机 `127.0.0.1`，**不要**改成 `0.0.0.0`（那会让局域网/公网所有人都能用你的号）。
- 内置限流（默认 12 秒/次）别调低，请求过密会被平台风控禁言/封号。

## 已验证 / 待验证

- ✅ `/health` `/v1/models` 实测返回正常。
- ✅ **真实 token 端到端完整对话已验证**（源码版 + exe 版均通过），PoW 被 DeepSeek 接受，正文完整返回。
- ✅ **SSE 增量聚合已修复**：DeepSeek 的私有 SSE 里"纯 v 字段追加行"此前被漏读，导致长回复截断成开头几个字；现已用真实抓包数据离线验证 + 真机验证（长句一字不差返回）。
- ✅ **exe 打包已验证**：首次运行自动生成 `config.yaml`、填 token 启动、完整对话，均通过。

## 安全默认

- 默认只绑 `127.0.0.1`，不暴露局域网。
- `config.yaml` 里 `api_key` 留空则不校验（仅本机自用）；填了则所有 `/v1/*` 接口要求 `Authorization: Bearer <key>`。
- 无状态设计：每次请求建会话、用完即删，不残留服务端历史。