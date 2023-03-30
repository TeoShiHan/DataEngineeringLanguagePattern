class Alternator:
    def __init__(self):
        self.num = 1
    
    def alternate(self):
        if self.num == 1:
            self.num = 2
        else: 
            self.num = 1
    
    def get_alternate(self):
        if self.num == 1:
            return 2
        else: 
            return 1