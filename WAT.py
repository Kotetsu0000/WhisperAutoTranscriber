import sys
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from queue import Queue
import threading
import traceback
import hashlib
import urllib
import ssl
import time

import whisper
from whisper import _MODELS, available_models
from whisper.audio import N_FRAMES
from whisper.model import Whisper, ModelDimensions
import soundcard as sc
import torch
import certifi
import numpy as np

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack(expand=1, fill=tk.BOTH, anchor=tk.NW)
        ##ここまでTkinterテンプレ##
        self.master=master

        #スピーカーリスト
        self.speaker_list = [str(i.name) for i in sc.all_speakers()]
        self.speaker_default = str(sc.default_speaker().name)

        self.q = Queue()

        #ウィジェットの生成
        self.create_widgets()

        self.load_thread = threading.Thread(target=self.set_model_auto, daemon=True)
        self.load_thread.start()

    def create_widgets(self):
        #全体領域の定義
        self.main_window = tk.PanedWindow(self, orient="vertical")
        self.main_window.pack(expand=True, fill=tk.BOTH, side=tk.TOP)

        #GPUが使用できるかの表示、モデルの準備完了に関しての表示
        self.set_window = tk.PanedWindow(self.main_window, orient="vertical")
        self.set_window.pack(fill=tk.BOTH, side=tk.TOP)

        #音声入力選択
        self.speaker_window = tk.PanedWindow(self.main_window, orient="vertical")
        self.speaker_window.pack(fill=tk.BOTH, side=tk.TOP)

        #ログ表示用領域
        self.log_window = tk.PanedWindow(self.main_window, orient="vertical")
        self.log_window.pack(fill=tk.BOTH, side=tk.TOP, expand=True)


        #GPUの使用ができるかのチェック
        self.GPU_ok = tk.BooleanVar()
        self.GPU_ok.set(torch.cuda.is_available())
        text = 'GPU使用可能' if torch.cuda.is_available() else 'GPU使用不可'
        self.GPU_check_box = tk.Checkbutton(self.set_window, variable=self.GPU_ok, state='disabled')
        self.GPU_check_box.pack(fill=tk.BOTH, side=tk.LEFT)
        self.GPU_use_label = ttk.Label(self.set_window, text=text)
        self.GPU_use_label.pack(fill=tk.BOTH, side=tk.LEFT)

        #モデルのロード確認
        self.model_ok = tk.BooleanVar()
        self.model_ok.set(False)
        self.model_check_box = tk.Checkbutton(self.set_window, variable=self.model_ok, state='disabled')
        self.model_check_box.pack(fill=tk.BOTH, side=tk.LEFT)
        self.model_use = tk.StringVar(value='モデルロード中...')
        self.model_use_label = ttk.Label(self.set_window, textvariable=self.model_use)
        self.model_use_label.pack(fill=tk.BOTH, side=tk.LEFT)

        #ログの削除ボタン
        self.log_delete_button = tk.Button(self.set_window, text='ログのクリア', command=self.clear_log)
        self.log_delete_button.pack(fill=tk.BOTH, side=tk.RIGHT)

        #モデルのロードに関して
        self.select_speaker = tk.StringVar(value=self.speaker_default)
        self.speaker_combobox = ttk.Combobox(self.speaker_window, textvariable= self.select_speaker, values=self.speaker_list, state="readonly")
        self.speaker_combobox.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        self.speaker_combobox.bind('<<ComboboxSelected>>', self.change_speaker)

        #ログ部分
        self.log_space = scrolledtext.ScrolledText(self.log_window, state='disabled')
        self.log_space.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    def change_speaker(self,event):
        self.change_speaker_bool = False

    def add_log(self, text):
        self.log_space.config(state='normal')
        self.log_space.insert(tk.END,text+'\n')
        self.log_space.config(state='disabled')
        self.log_space.see("end")

    def clear_log(self):
        self.log_space.config(state='normal')
        self.log_space.delete(0.,tk.END)
        self.log_space.config(state='disabled')

    def select_max_model(self):
        if torch.cuda.is_available():
            memory = torch.cuda.mem_get_info(0)[0]//1024**2
            if memory>9100:
                return 'large', 'cuda'
            elif memory>4772:
                return 'medium', 'cuda'
            elif memory>1888:
                return 'small', 'cuda'
            else:
                return 'base', 'cuda'
        else:
            return 'small', 'cpu'

    def set_model_auto(self):
        try:
            self.model_use.set('モデルサイズ計算中')
            self.model_size, self.device = self.select_max_model()
            if self.fetch_model(self.model_size):
                self.model_use.set(self.model_size+'モデル読み込み中')
                self.model = self.load_model(self.model_size, self.device)
            else:
                self.model_use.set(self.model_size+'モデルダウンロード中')
                while True:
                    if not(self.DL_window.master.winfo_exists()):
                        self.model_use.set(self.model_size+'モデル読み込み中')
                        self.model = self.load_model(self.model_size, self.device)
                        break
            self.model_use.set(self.model_size+'モデル起動中')
            self.model_ok.set(True)

            self.recording_thread = threading.Thread(target=self.recording, daemon=True)
            self.recognize_thread = threading.Thread(target=self.recognize, daemon=True)
            self.recording_thread.start()
            self.recognize_thread.start()
        except:
            messagebox.showerror('エラー', traceback.format_exc())
            self.master.destroy()

    def load_model(self, name, device):
        download_root = os.getenv(
            "XDG_CACHE_HOME", 
            os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        )
        target = os.path.join(download_root, os.path.basename(_MODELS[name]))
        with open(target, "rb") as fp:
            checkpoint = torch.load(fp, map_location=device)

        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)

    def fetch_model(self, name):
        download_root = os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))
        if name in _MODELS:
            target = os.path.join(download_root, os.path.basename(_MODELS[name]))
            expected_sha256 = _MODELS[name].split("/")[-2]
            #ファイルの有無を確認
            if os.path.isfile(target):
                with open(target, "rb") as f:
                    model_bytes = f.read()
                #ちゃんとしたファイルか確認
                if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
                    return True
                else:
                    #ダウンロードし直し
                    self.DL_model(name, download_root)
            else:
                #ないからダウンロード
                self.DL_model(name, download_root)
            return False
        else:
            messagebox.showerror('RuntimeError', f"Model {name} not found; available models = {available_models()}")
            self.master.destroy()

    def DL_model(self, name, download_root):
        if messagebox.askyesno('ダウンロード確認', f'{name}モデルをダウンロードしますか？'):
            root = tk.Toplevel()
            w = root.winfo_screenwidth()    #モニター横幅取得
            h = root.winfo_screenheight()   #モニター縦幅取得
            windw_w = 400
            windw_h = 100
            root.title(f"{name}モデルのダウンロード")
            root.geometry(str(windw_w)+"x"+str(windw_h)+"+"+str((w-windw_w)//2)+"+"+str((h-windw_h)//2))
            root.protocol("WM_DELETE_WINDOW", self.DL_exit)
            root.grab_set()
            root.resizable(width=False, height=False)
            self.DL_window = Download(master=root, url=_MODELS[name],path=download_root, del_func=self.DL_exit)
        else:
            self.master.destroy()

    def DL_exit(self, event=None, error=False):
        if error:
            self.DL_window.master.destroy()
            self.master.destroy()
        elif messagebox.askyesno('ダウンロードの中断', f'{self.model_size}モデルのダウンロードを中断しますか？'):
            self.DL_window.master.destroy()
            self.master.destroy()

    def recognize(self):
        if torch.cuda.is_available():
            options = whisper.DecodingOptions()
        else:
            options = whisper.DecodingOptions(fp16 = False)
        while True:
            if self.model_ok.get():
                audio = self.q.get()
                if (audio ** 2).max() > 0.0001:
                    mel = whisper.log_mel_spectrogram(audio)
                    if torch.cuda.is_available():
                        mel = whisper.pad_or_trim(mel, N_FRAMES).to(self.model.device).to(torch.float16)
                    else:
                        mel = whisper.pad_or_trim(mel, N_FRAMES).to(self.model.device)

                    # detect the spoken language
                    _, probs = self.model.detect_language(mel)

                    # decode the audio
                    result = whisper.decode(self.model, mel, options)

                    # print the recognized text
                    self.add_log(f'{max(probs, key=probs.get)}: {result.text}')
            else:time.sleep(1)

    def recording(self):
        SAMPLE_RATE = 16000
        INTERVAL = 3
        BUFFER_SIZE = 4096
        b = np.ones(100) / 100

        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        n = 0
        while True:
            self.change_speaker_bool=True
            with sc.get_microphone(id=str(self.select_speaker.get()), include_loopback=True).recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
                while self.change_speaker_bool:
                    while n < SAMPLE_RATE * INTERVAL:
                        data = mic.record(BUFFER_SIZE)
                        audio[n:n+len(data)] = data.reshape(-1)
                        n += len(data)

                    # find silent periods
                    m = n * 4 // 5
                    vol = np.convolve(audio[m:n] ** 2, b, 'same')
                    m += vol.argmin()
                    self.q.put(audio[:m])

                    audio_prev = audio
                    audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
                    audio[:n-m] = audio_prev[m:n]
                    n = n-m


class Download(tk.Frame):
    def __init__(self, url, path, master=None, del_func=None):
        tk.Frame.__init__(self, master)
        self.pack(expand=1, fill=tk.BOTH, anchor=tk.NW)
        ##ここまでTkinterテンプレ##
        self.master=master
        self.del_func = del_func

        self.url = url
        self.path = path
        self.source = urllib.request.urlopen(self.url, context=ssl.create_default_context(cafile=certifi.where()))

        self.create_widgets()
        self.DL_thread = threading.Thread(target=self._download, daemon=True)
        self.DL_thread.start()

    def create_widgets(self):
        self.label = ttk.Label(self, text='モデルのダウンロード中')
        self.label.pack(side=tk.TOP, pady=5)

        self.dl_prog_window = tk.PanedWindow(self, orient="vertical")
        self.dl_prog_window.pack(fill=tk.BOTH, side=tk.TOP)

        self.dl_prog_percent = tk.StringVar(value='0%')
        self.dl_prog_percent_label = ttk.Label(self.dl_prog_window, textvariable=self.dl_prog_percent)
        self.dl_prog_percent_label.pack(fill=tk.BOTH, side=tk.RIGHT, padx=5, pady=5)

        self.var = tk.IntVar()
        self.dl_prog = ttk.Progressbar(self.dl_prog_window,maximum=self.source.info().get("Content-Length"),mode="determinate",variable=self.var, style="bar.Horizontal.TProgressbar")
        self.dl_prog.pack(expand=True, fill=tk.BOTH, side=tk.LEFT, padx=5, pady=5)

        self.log_delete_button = tk.Button(self, text='キャンセル', command=self.del_func)
        self.log_delete_button.pack(side=tk.TOP, pady=5)

    def _download(self):
        url, root = self.url, self.path
        os.makedirs(root, exist_ok=True)

        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(root, os.path.basename(url))

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            messagebox.showerror('RuntimeError',f"{download_target} exists and is not a regular file")
            self.del_func(error=True)

        with self.source as source, open(download_target, "wb") as output:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                self.var.set(self.var.get()+len(buffer))
                percent = self.var.get()/int(self.source.info().get("Content-Length"))*100
                self.dl_prog_percent.set(str(int(percent))+'%')

        model_bytes = open(download_target, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
            messagebox.showerror('RuntimeError',f"Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model.")
            self.del_func(error=True)

        self.master.destroy()#無事終了


def temp_path(relative_path):
    try:
        #Retrieve Temp Path
        base_path = sys._MEIPASS
    except Exception:
        #Retrieve Current Path Then Error 
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    root = tk.Tk()
    w = root.winfo_screenwidth()    #モニター横幅取得
    h = root.winfo_screenheight()   #モニター縦幅取得
    windw_w = 700
    windw_h = 400

    root.title(u"Whisper Auto Transcriber")
    iconfile = 'icon.ico'
    icon = temp_path(iconfile)
    root.iconbitmap(default=icon)
    root.geometry(str(windw_w)+"x"+str(windw_h)+"+"+str((w-windw_w)//2)+"+"+str((h-windw_h)//2))
    app = Application(master=root)
    app.mainloop()
