from queue import PriorityQueue

class test():
    def __init__(self, pq):
        self.pq = pq
        self.edit_list(self.pq)

    def edit_list(self, pq):
        pq.put((0, 'it works'))

pq = PriorityQueue()
pq.put((1, 'idk if it works'))
pq_editor = test(pq)
print(pq.queue)
