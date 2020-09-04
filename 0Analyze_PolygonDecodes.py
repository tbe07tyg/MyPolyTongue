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
x = np.linspace(0.1, 20, 100)
# scatter plot
tips = sns.load_dataset("tips")
ax = sns.lineplot(x=x, y=my_IoU_data_polydeocde[...,0], ci=my_IoU_data_polydeocde[...,1])
ax.errorbar(x, my_IoU_data_polydeocde[...,0], yerr=my_IoU_data_polydeocde[...,1], fmt='-o', label="our") #fmt=None to plot bars only
ax2 = sns.lineplot(x=x, y=their_IoU_data_polydeocde[...,0], ci=their_IoU_data_polydeocde[...,1])
ax2.errorbar(x, their_IoU_data_polydeocde[...,0], yerr=their_IoU_data_polydeocde[...,1], fmt='-x', label="their") #fmt=None to plot bars only


# sns.lineplot(x=x, y=my_IoU_data_polydeocde[...,0])
plt.title("Compare Polar Conversion Methods")
plt.xlabel("Angle Step")
plt.ylabel("IoU")
plt.legend(loc="lower right")
plt.savefig("ComparePolarConversionMethods.jpg", dpi=300)
plt.show()