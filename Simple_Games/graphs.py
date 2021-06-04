import matplotlib.pyplot as plt
import seaborn as sns
import ast

compass_file = open("compasses/compass_conv_per.txt", "r")
lines = compass_file.readlines()


graph = []
for k in range(0, 10):
    graph.extend(ast.literal_eval(lines[k]))
plt.figure(figsize=(5, 5))
# sns.kdeplot(graph , bw = 0.5 , fill = True)
sns.distplot(graph)
plt.show()


i = 0
graph = []
for line in lines:
    if i % 50 != 49:
        graph.extend(ast.literal_eval(line))
    else:
        plt.figure(figsize=(5, 5))
        # sns.kdeplot(graph , bw = 0.5 , fill = True)
        sns.distplot(graph)
        plt.show()

        graph = ast.literal_eval(line)
    i += 1


# plt.figure(figsize = (5,5))
# # sns.kdeplot(graph , bw = 0.5 , fill = True)
# sns.distplot(graph)
# plt.show()

# Iterate through the five airlines
# for i in range(1, len(segments)):
#     # Subset of the airline:
#     subset = compass[segments[i-1]:segments[i]]
#
#     # Draw the density plot
#     sns.distplot(subset['arr_delay'], hist=False, kde=True,
#                  kde_kws={'linewidth': 3},
#                  label=str(segments[i]))
#
# # Plot formatting
# plt.legend(prop={'size': 16}, title='Iterations')
# plt.title('Density Plot at different episodes')
# plt.xlabel('Angle')
# plt.ylabel('Density')