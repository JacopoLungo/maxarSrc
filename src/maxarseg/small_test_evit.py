import rasterio
from rasterio.windows import Window


def main():
    file_path = "path/to/your/tif/file.tif"
    window = Window(col_off=500, row_off=500, width=1024, height=1024)
    
    with rasterio.open(file_path) as src:
        data = src.read(window=window)
    
    
    
    
    print(data)


    


if __name__ == "__main__":
    main()
