


class Logger(object):

    children: [any] = []

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Logger, cls).__new__(cls)
        return cls.instance

    def subscribe(self, subscriber: any):
        self.children.append(subscriber)

    def log(self, patientId: str, message: str):
        for child in self.children:
            child(patientId, message)

