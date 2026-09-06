from graphviz import Digraph
import math

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data= {self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out

    def tanh(self):
        x = self.data
        t =(math.exp(2*x) -1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            if v._prev:
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(uid, label = "{%s |data %.4f | grad %.4f}"%(n.label, n.data, n.grad), shape='record')

        if n._op:
            op_uid = uid + n._op
            dot.node(op_uid, label=n._op)
            dot.edge(op_uid, uid)

    for n1, n2 in edges:
        op_uid = str(id(n2)) + n2._op
        dot.edge(str(id(n1)), op_uid)

    return dot

#Inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

#Weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

#Bias
b = Value(6.8813735870195432, label='b')

#Weighted Sum
x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b; n.label = 'n'

o = n.tanh()
o.grad = 1.0
n.grad = 0.5 #1- tanh(x)**2
x1w1x2w2.grad = 0.5
b.grad = 0.5
x1w1.grad = 0.5
x2w2.grad = 0.5
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad
dot = draw_dot(o); o.label = 'o'
dot.render('graph', view=False, cleanup=True)   