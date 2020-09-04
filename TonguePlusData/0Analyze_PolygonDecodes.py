# if decode == "my":
#     with open('my_IoU_data_polydeocde', 'wb') as fp:
#         pickle.dump(my_IoU_data, fp)
# else:
#     with open('their_IoU_data_polydeocde', 'wb') as fp:
#         pickle.dump(their_IoU_data, fp)
import pickle

with open ('my_IoU_data_polydeocde', 'rb') as fp:
    my_IoU_data_polydeocde = pickle.load(fp)

with open ('their_IoU_data_polydeocde', 'rb') as fp:
    their_IoU_data_polydeocde = pickle.load(fp)

print("my_IoU_data_polydeocde:", my_IoU_data_polydeocde)