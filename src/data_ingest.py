from abc import ABC, abstractmethod
import pandas as pd
import os
import zipfile

class DataIngestor(ABC):

    """
    class / interface untuk membuat data ingestor
    """

    @abstractmethod
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        pass

class ZipDataIngestor(DataIngestor):
    """
    class untuk ingest file berekstensi .zip
    """
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        # cek ekstensi fil
        if not file_path.endswith(".zip"):
            return ValueError("The provided file is not a .zip file.")
        
        # buka file zip dan ekstrak filenya ke dalam folder extracted data
        with zipfile.ZipFile(file_path) as zip_ref:
            zip_ref.extractall("dataset/extracted_data")
        
        # setelah di ekstrak, cek apakah ada file dan memiliki ekstensi .csv 
        # kalau ada masukan kedalam variabel list csv_files
        list_files = os.listdir("dataset/extracted_data")
        csv_files = [file for file in list_files if file.endswith(".zip")]

        # ambil alamat file csv
        extracted_csv_path = os.path.join("dataset/extracted_data", list_files[0])
        # definisikan nama file baru
        target_csv_path = os.path.join("dataset/extracted_data", "Telco-Customer-Churn.csv")

        # ubah nama file
        if extracted_csv_path != target_csv_path:
            os.rename(extracted_csv_path, target_csv_path)


        # kalau di dalam folder tidak ada file, munculkan error
        if len(csv_files) == 0:
            return FileNotFoundError("No csv file found in the extracted data folder")
        
        # kalau di dalam folder terdapat lebih dari 1 file, munculkan error
        if len(csv_files) > 1:
            return ValueError("Multiple csv file found. Please choose spesify file")
        
        # kalau terdapat 1 file .csv
        # masukan data ke dalam bentuk dataframe
        csv_file_path = os.path.join("dataset/extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)
        return df

class CsvDataIngestor(DataIngestor):
    """
    class untuk ingest data yang memiliki ekstensi .csv
    """

    def ingest_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        return df
    
class DataIngestorFactory:
    """
    class untuk menentukan data ingestor mana yang digunakan
    """
    def get_data_ingestor(self, file_ekstension: str) -> DataIngestor:
        if file_ekstension == '.zip':
            return ZipDataIngestor()
        if file_ekstension == '.csv':
            return CsvDataIngestor
        else:
            print(f"No ingestor available for file extension {file_ekstension}")


if __name__ == "__main__":
    zip_ingestor = DataIngestorFactory()
    zip_ingest = zip_ingestor.get_data_ingestor(".zip")
    zip_ingest.ingest_data("dataset/raw/Telco-Customer-Churn.zip")