# @Author       : Duhongkai
# @Time         : 2024/1/2 15:11
# @Description  : 发送邮件通知

import smtplib
from email.mime.text import MIMEText
from email.header import Header


def send_msg(subject, body) -> None:
    # 构建邮件
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = ''
    msg['To'] = ''

    # 发送邮件
    smtp_server = ''
    smtp_port = 587
    sender_email = ''
    password = ''  # 在QQ邮箱设置里拿到的码

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, [msg['To']], msg.as_string())
        print('邮件发送成功')
    except smtplib.SMTPException as e:
        print('邮件发送失败:', str(e))
