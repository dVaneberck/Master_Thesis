import matplotlib.pyplot as plt
import seaborn as sns
import ast

compass_file = open("compasses/compass_small_dumb.txt", "r")
lines = compass_file.readlines()

rewards_file = open("rewards/rewards_small_dumb.txt")
r_lines = rewards_file.readlines()
x = []
y = []
for l in r_lines:
    a, b = l.split(" ")
    x.append(int(a))
    y.append(float(b))

graph = []
for k in range(0, 1):
    graph.extend(ast.literal_eval(lines[k]))
plt.figure(figsize=(5, 5))
plt.hist(graph)
plt.xlabel('angle (°)')
plt.show()

i = 0
graph = []
for line in lines:
    if i % 21 != 20:
        graph.extend(ast.literal_eval(line))
    else:
        plt.figure(figsize=(5, 5))
        # sns.kdeplot(graph , bw = 0.5 , fill = True)
        # sns.distplot(graph)
        plt.hist(graph)
        plt.xlabel('angle (°)')
        plt.show()

        graph = ast.literal_eval(line)
    i += 1


plt.figure(figsize=(5, 5))
plt.plot(x, y, 'bo')
plt.axhline(y=100, color='r', linestyle='-')
plt.axhline(y=0, color='k')
plt.xlabel('episodes')
plt.ylabel('cumulative reward')
plt.show()
