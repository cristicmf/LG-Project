#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件工具模块
"""

import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header

# 发送邮件函数
def send_email(smtp_config, recipient_email, subject, body):
    """使用 SMTP 协议发送邮件"""
    try:
        # 从配置文件中读取 SMTP 服务器配置
        smtp_server = smtp_config["server"]
        smtp_port = smtp_config["port"]
        smtp_username = smtp_config["username"]
        smtp_password = smtp_config["password"]
        sender_email = smtp_config["sender_email"]
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # 添加邮件正文
        msg.attach(MIMEText(body, 'plain'))
        
        # 连接到 SMTP 服务器并发送邮件
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        return True, f"Email sent to {recipient_email}"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# 接收邮件函数
def receive_emails(imap_config):
    """使用 IMAP 协议接收邮件"""
    try:
        # 从配置文件中读取 IMAP 服务器配置
        imap_server = imap_config["server"]
        imap_username = imap_config["username"]
        imap_password = imap_config["password"]
        
        # 连接到 IMAP 服务器并选择收件箱
        with imaplib.IMAP4_SSL(imap_server) as imap:
            imap.login(imap_username, imap_password)
            status, messages_count = imap.select("INBOX")
            
            # 搜索未读邮件
            status, messages_ids = imap.search(None, "UNSEEN")
            
            # 处理每封邮件
            emails = []
            for msg_id in messages_ids[0].split():
                status, msg_data = imap.fetch(msg_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        # 解析邮件
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding if encoding else "utf-8")
                        
                        from_email = msg.get("From")
                        
                        # 获取邮件正文
                        if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                content_disposition = str(part.get("Content-Disposition"))
                                if content_type == "text/plain" and "attachment" not in content_disposition:
                                    body = part.get_payload(decode=True).decode()
                                    break
                        else:
                            body = msg.get_payload(decode=True).decode()
                        
                        # 添加邮件到结果列表
                        emails.append({
                            "subject": subject,
                            "from_email": from_email,
                            "body": body
                        })
            
            return True, emails
    except Exception as e:
        return False, f"Failed to receive email: {str(e)}"
