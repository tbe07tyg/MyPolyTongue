# if decode == "my":
#     with open('my_IoU_data_polydeocde', 'wb') as fp:
#         pickle.dump(my_IoU_data, fp)
# else:
#     with open('their_IoU_data_polydeocde', 'wb') as fp:
#         pickle.dump(their_IoU_data, fp)
import pickle
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
with open ('my_IoU_data_polydeocde', 'rb') as fp:
    my_IoU_data_polydeocde = np.array(pickle.load(fp))

with open ('their_IoU_data_polydeocde', 'rb') as fp:
    their_IoU_data_polydeocde =  np.array(pickle.load(fp))

print("my_IoU_data_polydeocde:", my_IoU_data_polydeocde.shape)
print(my_IoU_data_polydeocde)
# find the maximum value position of iou:
their_largest = np.argmax(their_IoU_data_polydeocde[...,0])
print("their largest position:", their_largest)
print("their highest iou:", their_largest,their_IoU_data_polydeocde[...,0][their_largest])

our_max_index =  np.argmax(my_IoU_data_polydeocde[...,0])

result_as_theirmax_index = idx = (np.abs(my_IoU_data_polydeocde[...,0] - their_IoU_data_polydeocde[...,0][their_largest])).argmin()
print("result_closeto_theirmax_index", result_as_theirmax_index)
print("fair iou", my_IoU_data_polydeocde[...,0][result_as_theirmax_index])
x = np.linspace(0.1, 40, 100)
# scatter plot
tips = sns.load_dataset("tips")
ax = sns.lineplot(x=x, y=my_IoU_data_polydeocde[...,0], ci=my_IoU_data_polydeocde[...,1])
ax.errorbar(x, my_IoU_data_polydeocde[...,0], yerr=my_IoU_data_polydeocde[...,1], fmt='-o', label="Our", color='r') #fmt=None to plot bars only
ax2 = sns.lineplot(x=x, y=their_IoU_data_polydeocde[...,0], ci=their_IoU_data_polydeocde[...,1])
ax2.errorbar(x, their_IoU_data_polydeocde[...,0], yerr=their_IoU_data_polydeocde[...,1], fmt='-x', label="Poly-yolo", color='g') #fmt=None to plot bars only



# sns.lineplot(x=x, y=my_IoU_data_polydeocde[...,0])
plt.title("Compare Polar Conversion Methods", fontweight='bold')
plt.xlabel("Angle Step", fontweight='bold')
plt.ylabel("IoU", fontweight='bold')
plt.xlim([0.1, 40])
plt.ylim([0, 1])
plt.legend(loc="lower right", prop={'size': 15})


plt.annotate("max({:.2f}, {:.2f})" .format(x[their_largest],their_IoU_data_polydeocde[...,0][their_largest] ),
             xy=(x[their_largest],their_IoU_data_polydeocde[...,0][their_largest]),
             xytext=(50, -100), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05))

plt.annotate("max({:.2f}, {:.2f})" .format(x[our_max_index],my_IoU_data_polydeocde[...,0][our_max_index] ),
             xy=(x[our_max_index],my_IoU_data_polydeocde[...,0][our_max_index]),
             xytext=(50, -100), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05))

plt.annotate("fair({:.2f}, {:.2f})" .format(x[result_as_theirmax_index],my_IoU_data_polydeocde[...,0][result_as_theirmax_index] ),
             xy=(x[result_as_theirmax_index],my_IoU_data_polydeocde[...,0][result_as_theirmax_index]),
             xytext=(40, -150), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05))

plt.savefig("ComparePolarConversionMethods.jpg", dpi=300)
plt.show()