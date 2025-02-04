class QuickFeedbackException(Exception):
    def __init__(self, message=None, data=None):
        super().__init__(message)
        self.data = data