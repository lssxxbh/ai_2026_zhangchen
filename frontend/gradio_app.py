import gradio as gr
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

API_BASE = "http://localhost:8000/api"

current_token: Optional[str] = None
current_user_id: Optional[int] = None
current_conversation_id: Optional[int] = None


def create_gradio_app():
    with gr.Blocks(title="智能体检报告解析系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏥 智能体检报告解析系统")

        # 状态变量
        auth_token = gr.State(None)
        user_id_state = gr.State(None)
        conversation_id_state = gr.State(None)
        conversations_cache = gr.State([])  # 格式: [{"id": 1, "title": "...", "created_at": "..."}, ...]
        logged_in = gr.State(False)

        # 登录页面
        with gr.Row(visible=True) as login_row:
            with gr.Column(scale=1):
                gr.Markdown("## 用户认证")
                with gr.Tab("登录"):
                    login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                    login_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                    login_btn = gr.Button("登录", variant="primary")
                    login_msg = gr.Textbox(label="", interactive=False)

                with gr.Tab("注册"):
                    reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                    reg_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                    reg_btn = gr.Button("注册", variant="primary")
                    reg_msg = gr.Textbox(label="", interactive=False)

        # 聊天页面
        with gr.Row(visible=False) as chat_row:
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("## 📋 会话")
                new_conversation_btn = gr.Button("➕ 新建会话", variant="primary")
                conversations_dropdown = gr.Dropdown(
                    label="选择会话",
                    choices=[],
                    interactive=True,
                    allow_custom_value=False
                )
                refresh_conversations_btn = gr.Button("🔄 刷新")
                delete_conversation_btn = gr.Button("🗑️ 删除会话", variant="stop")
                gr.Markdown("---")
                logout_btn = gr.Button("🚪 退出登录", variant="secondary")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=MedicalAI"),
                    bubble_full_width=False
                )

                with gr.Row():
                    file_input = gr.File(
                        label="上传文件 (PDF/图片/TXT)",
                        file_count="single",
                        file_types=["pdf", "png", "jpg", "jpeg", "bmp", "txt"]
                    )
                with gr.Row():
                    text_input = gr.Textbox(
                        label="输入消息",
                        placeholder="请输入文本或上传文件...",
                        scale=4,
                        lines=2
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                json_output = gr.Code(
                    label="解析结果 (JSON)",
                    language="json",
                    interactive=False
                )

        # 处理登录
        def handle_login(username: str, password: str):
            global current_token, current_user_id
            try:
                response = requests.post(
                    f"{API_BASE}/login",
                    json={"username": username, "password": password}
                )
                data = response.json()

                if data.get("code") == 0:
                    token_data = data["data"]
                    current_token = token_data["access_token"]
                    current_user_id = token_data["user_id"]
                    convs = load_conversations_list(current_token)
                    dropdown_choices = format_conversations_for_dropdown(convs)
                    return (
                        current_token,
                        current_user_id,
                        f"登录成功: {username}",
                        gr.update(visible=False),
                        gr.update(visible=True),
                        True,
                        "",
                        convs,
                        None,
                        gr.update(choices=dropdown_choices, value=None)
                    )
                else:
                    return None, None, data.get("msg", "登录失败"), gr.update(), gr.update(), False, "", [], None, gr.update(choices=[], value=None)
            except Exception as e:
                return None, None, f"错误: {str(e)}", gr.update(), gr.update(), False, "", [], None, gr.update(choices=[], value=None)

        # 处理注册
        def handle_register(username: str, password: str):
            global current_token, current_user_id
            try:
                response = requests.post(
                    f"{API_BASE}/register",
                    json={"username": username, "password": password}
                )
                data = response.json()

                if data.get("code") == 0:
                    token_data = data["data"]
                    current_token = token_data["access_token"]
                    current_user_id = token_data["user_id"]
                    convs = load_conversations_list(current_token)
                    dropdown_choices = format_conversations_for_dropdown(convs)
                    return (
                        current_token,
                        current_user_id,
                        f"注册成功: {username}",
                        gr.update(visible=False),
                        gr.update(visible=True),
                        True,
                        "",
                        convs,
                        None,
                        gr.update(choices=dropdown_choices, value=None)
                    )
                else:
                    return None, None, data.get("msg", "注册失败"), gr.update(), gr.update(), False, "", [], None, gr.update(choices=[], value=None)
            except Exception as e:
                return None, None, f"错误: {str(e)}", gr.update(), gr.update(), False, "", [], None, gr.update(choices=[], value=None)

        # 加载会话列表
        def load_conversations_list(token: Optional[str]):
            if not token:
                return []
            try:
                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(f"{API_BASE}/conversations", headers=headers)
                data = response.json()
                if data.get("code") == 0:
                    return data.get("data", [])
                return []
            except Exception:
                return []

        # 将会话列表格式化为dropdown选项
        def format_conversations_for_dropdown(conversations: List[Dict]):
            # 返回格式: [(label, value), ...]
            return [
                (f"{conv['title']} (ID: {conv['id']})", conv['id'])
                for conv in conversations
            ]

        # 退出登录
        def handle_logout():
            global current_token, current_user_id, current_conversation_id
            current_token = None
            current_user_id = None
            current_conversation_id = None
            return (
                None,
                None,
                gr.update(visible=True),
                gr.update(visible=False),
                False,
                [],
                "",
                None,
                gr.update(choices=[], value=None)
            )

        # 选择会话
        def select_conversation(selected_conv_id: Optional[int], token: Optional[str], conv_cache: List):
            global current_conversation_id
            if not token or not selected_conv_id:
                return [], "", None

            try:
                conv_id = selected_conv_id
                current_conversation_id = conv_id

                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(f"{API_BASE}/history/{conv_id}", headers=headers)
                data = response.json()

                if data.get("code") == 0:
                    messages = data.get("data", [])
                    new_history = []
                    last_json = ""

                    i = 0
                    while i < len(messages):
                        msg = messages[i]
                        role = msg["role"]
                        content = msg["message"]

                        if role in ["user", "USER"]:
                            new_history.append((content, None))

                            if i + 1 < len(messages):
                                next_msg = messages[i + 1]
                                next_role = next_msg["role"]
                                if next_role in ["assistant", "ASSISTANT"]:
                                    new_history[-1] = (content, next_msg["message"])
                                    if next_msg.get("json_result"):
                                        last_json = json.dumps(next_msg["json_result"], ensure_ascii=False, indent=2)
                                    i += 1
                        elif role in ["assistant", "ASSISTANT"]:
                            if len(new_history) > 0:
                                if new_history[-1][1] is None:
                                    new_history[-1] = (new_history[-1][0], content)
                                else:
                                    new_history.append(("", content))
                            else:
                                new_history.append(("", content))

                            if msg.get("json_result"):
                                last_json = json.dumps(msg["json_result"], ensure_ascii=False, indent=2)

                        i += 1

                    return new_history, last_json, conv_id
            except Exception as e:
                print(f"Select conversation error: {e}")
                pass

            return [], "", None

        # 新建会话
        def new_conversation(token: Optional[str]):
            global current_conversation_id
            current_conversation_id = None
            convs = load_conversations_list(token)
            dropdown_choices = format_conversations_for_dropdown(convs)
            return [], "", None, convs, gr.update(choices=dropdown_choices, value=None)

        # 删除会话
        def delete_conversation(token: Optional[str], conv_id: Optional[int]):
            global current_conversation_id
            if not token or not conv_id:
                convs = load_conversations_list(token)
                dropdown_choices = format_conversations_for_dropdown(convs)
                return [], "", None, convs, gr.update(choices=dropdown_choices, value=None)

            try:
                headers = {"Authorization": f"Bearer {token}"}
                requests.delete(f"{API_BASE}/conversation/{conv_id}", headers=headers)
            except Exception:
                pass

            current_conversation_id = None
            convs = load_conversations_list(token)
            dropdown_choices = format_conversations_for_dropdown(convs)
            return [], "", None, convs, gr.update(choices=dropdown_choices, value=None)

        # 刷新会话列表
        def refresh_conversations(token: Optional[str]):
            convs = load_conversations_list(token)
            dropdown_choices = format_conversations_for_dropdown(convs)
            return convs, gr.update(choices=dropdown_choices)

        # 获取文件名，兼容不同类型的文件对象
        def get_file_name(file):
            if not file:
                return None
            if isinstance(file, dict):
                return file.get('name')
            if hasattr(file, 'name'):
                return file.name
            if isinstance(file, tuple) and len(file) > 0:
                return get_file_name(file[0])
            return str(file)

        # 获取文件路径，兼容不同类型的文件对象
        def get_file_path(file):
            if not file:
                return None
            if isinstance(file, dict):
                return file.get('path')
            if hasattr(file, 'path'):
                return file.path
            if isinstance(file, tuple) and len(file) > 0:
                return get_file_path(file[0])
            return None

        # 发送消息
        def send_message(
            text: str,
            file,
            token: Optional[str],
            conv_id: Optional[int],
            history: List
        ):
            global current_conversation_id

            if not token:
                convs = load_conversations_list(token)
                dropdown_choices = format_conversations_for_dropdown(convs)
                return history, "", conv_id, "", None, convs, gr.update(choices=dropdown_choices)

            user_text = text or ""
            file_display_name = get_file_name(file)
            file_path = get_file_path(file)
            
            # 构建显示消息 - 确保文件名正确显示
            display_parts = []
            if user_text:
                display_parts.append(user_text)
            if file_display_name:
                display_parts.append(f"[文件: {file_display_name}]")
            
            display_text = "\n".join(display_parts) if display_parts else ""
            history.append((display_text, None))

            try:
                headers = {"Authorization": f"Bearer {token}"}
                files = {}
                data = {"text": user_text}

                if conv_id:
                    data["conversation_id"] = conv_id

                if file_path:
                    files["file"] = (Path(file_display_name).name, open(file_path, "rb"), "application/octet-stream")
                elif file and hasattr(file, 'read'):
                    files["file"] = (file_display_name, file, "application/octet-stream")

                response = requests.post(
                    f"{API_BASE}/chat",
                    headers=headers,
                    data=data,
                    files=files if file else None
                )
                result = response.json()

                if result.get("code") == 0:
                    chat_data = result["data"]
                    current_conversation_id = chat_data["conversation_id"]
                    json_result = json.dumps(chat_data["result"], ensure_ascii=False, indent=2)

                    history[-1] = (display_text, "解析完成！请查看下方JSON结果。")
                    convs = load_conversations_list(token)
                    dropdown_choices = format_conversations_for_dropdown(convs)
                    return history, json_result, current_conversation_id, "", None, convs, gr.update(choices=dropdown_choices, value=current_conversation_id)
                else:
                    history[-1] = (display_text, f"错误: {result.get('msg')}")
                    convs = load_conversations_list(token)
                    dropdown_choices = format_conversations_for_dropdown(convs)
                    return history, "", conv_id, "", None, convs, gr.update(choices=dropdown_choices)
            except Exception as e:
                history[-1] = (display_text, f"错误: {str(e)}")
                convs = load_conversations_list(token)
                dropdown_choices = format_conversations_for_dropdown(convs)
                return history, "", conv_id, "", None, convs, gr.update(choices=dropdown_choices)

        # 事件绑定 - 登录/注册
        login_btn.click(
            handle_login,
            inputs=[login_username, login_password],
            outputs=[
                auth_token,
                user_id_state,
                login_msg,
                login_row,
                chat_row,
                logged_in,
                conversations_cache,
                conversation_id_state,
                conversations_dropdown
            ]
        )

        reg_btn.click(
            handle_register,
            inputs=[reg_username, reg_password],
            outputs=[
                auth_token,
                user_id_state,
                reg_msg,
                login_row,
                chat_row,
                logged_in,
                conversations_cache,
                conversation_id_state,
                conversations_dropdown
            ]
        )

        logout_btn.click(
            handle_logout,
            outputs=[
                auth_token,
                user_id_state,
                login_row,
                chat_row,
                logged_in,
                chatbot,
                json_output,
                conversation_id_state,
                conversations_dropdown
            ]
        )

        # 事件绑定 - 聊天
        refresh_conversations_btn.click(
            refresh_conversations,
            inputs=[auth_token],
            outputs=[conversations_cache, conversations_dropdown]
        )

        conversations_dropdown.change(
            select_conversation,
            inputs=[conversations_dropdown, auth_token, conversations_cache],
            outputs=[chatbot, json_output, conversation_id_state]
        )

        new_conversation_btn.click(
            new_conversation,
            inputs=[auth_token],
            outputs=[chatbot, json_output, conversation_id_state, conversations_cache, conversations_dropdown]
        )

        delete_conversation_btn.click(
            delete_conversation,
            inputs=[auth_token, conversation_id_state],
            outputs=[chatbot, json_output, conversation_id_state, conversations_cache, conversations_dropdown]
        )

        submit_btn.click(
            send_message,
            inputs=[text_input, file_input, auth_token, conversation_id_state, chatbot],
            outputs=[chatbot, json_output, conversation_id_state, text_input, file_input, conversations_cache, conversations_dropdown]
        )

        text_input.submit(
            send_message,
            inputs=[text_input, file_input, auth_token, conversation_id_state, chatbot],
            outputs=[chatbot, json_output, conversation_id_state, text_input, file_input, conversations_cache, conversations_dropdown]
        )

    return demo


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)

