from FOTS.data_loader import *

if __name__ == "__main__":
    ds = ICDAR(r"C:\Users\ajava\Projekte\DataSets\ICDAR 2015\4.4\train")
    a = ds[0]