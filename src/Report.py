import datetime
import pickle
import os
import pandas as pd

class ReportData:
    """
    Simple data class for report entries.
    """
    def __init__(self, genre: str, mood: str, date: datetime.datetime = None):
        self.genre = genre
        self.mood = mood
        self.date = date or datetime.datetime.now()

    def to_dict(self):
        return {
            'Genre': self.genre,
            'Mood': self.mood,
            'Date': self.date
        }


def updateData(genre: str, mood: str):
    """
    Append a new ReportData entry to data.pkl.
    """
    file_path = "data.pkl"
    rd = ReportData(genre, mood)

    # Load existing list or start fresh
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'rb') as pfile:
                data_list = pickle.load(pfile)
                if not isinstance(data_list, list):
                    data_list = [data_list]
        except Exception:
            data_list = []
    else:
        data_list = []

    # Append and save back
    data_list.append(rd)
    with open(file_path, 'wb') as pfile:
        pickle.dump(data_list, pfile)


def getData():
    """
    Read data.pkl and display summary counts and table.
    """
    file_path = "data.pkl"
    if not os.path.isfile(file_path):
        print("No data")
        return

    try:
        with open(file_path, 'rb') as pfile:
            data_list = pickle.load(pfile)
            if not isinstance(data_list, list):
                data_list = [data_list]
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Convert to DataFrame
    df = pd.DataFrame([entry.to_dict() for entry in data_list])

    # Summary
    total = len(df)
    print(f"Total: {total}")
    print("===================")
    print("Genres")
    print("===================")
    print(df.groupby('Genre').size().reset_index(name='Amount'))

    print("===================")
    print("Moods")
    print("===================")
    print(df.groupby('Mood').size().reset_index(name='Amount'))

    print("===================")
    print("All Entries (most recent first)")
    print("===================")
    print(df.sort_values(by='Date', ascending=False))
