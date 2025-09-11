# @Author       : Duhongkai
# @Time         : 2024/1/16 11:36
# @Description  : 响应数据
import datetime


class ResponseData(object):
    def __init__(self, status="200", message="success", data=None):
        self.status = status
        self.message = message
        self.data = data
        self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {"status": self.status,
                "message": self.message,
                "data": self.data,
                "date": self.date}

