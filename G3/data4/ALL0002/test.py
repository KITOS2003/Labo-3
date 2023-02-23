with open("F0002TEK.SET", "r") as file:
    config = file.read()

aux = config[config.find("CH1"):]
aux = aux[aux.find("SCALE"):]
aux = aux[:aux.find(";")]
aux = aux.replace("SCALE ", "")

base = aux[:aux.find("E")]
exp = aux[aux.find("E"):].replace("E", "")

scale = base * 10**exp

print("base:", base)
print("exp:", exp)
