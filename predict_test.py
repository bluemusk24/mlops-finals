import pickle

with open("model_SVC.bin", "rb") as f_in:
    (dv, model_SVC) = pickle.load(f_in)


landmine = {"voltage": 0.341389, "height": 0.818182, "soil_types": 0.4}


X = dv.transform(landmine)
pred = model_SVC.predict(X)[0]


print("input", landmine)
print("landmine_type =", pred)


if pred == 0:
    print("landmine type is not detected")
elif pred == 1:
    print("landmine type is Anti-Tank")
elif pred == 2:
    print("landmine type is Anti-Personnel")
elif pred == 3:
    print("landmine type is Booby Trapped Anti-Personnel")
elif pred == 4:
    print("landmine type is M14 Anti-Personnel")
else:
    print("nothing")
