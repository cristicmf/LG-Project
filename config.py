# 邮件服务器配置

# SMTP 服务器配置
SMTP_CONFIG = {
    "server": "smtp.example.com",  # 修改为实际的 SMTP 服务器地址
    "port": 587,  # SMTP 服务器端口，通常为 587
    "username": "your-email@example.com",  # 修改为实际的邮箱用户名
    "password": "your-password",  # 修改为实际的邮箱密码
    "sender_email": "support@example.com"  # 修改为实际的发件人邮箱
}

# IMAP 服务器配置
IMAP_CONFIG = {
    "server": "imap.example.com",  # 修改为实际的 IMAP 服务器地址
    "username": "your-email@example.com",  # 修改为实际的邮箱用户名
    "password": "your-password"  # 修改为实际的邮箱密码
}

# 应用配置
APP_CONFIG = {
    "email_archive_dir": "email_archive",  # 邮件归档目录
    "max_search_results": 3  # 最大搜索结果数
}
