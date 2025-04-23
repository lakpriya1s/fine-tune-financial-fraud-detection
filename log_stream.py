import io
class StreamToStreamlit(io.StringIO):
    def __init__(self, output_widget, progress_bar=None):
        super().__init__()
        self.output_widget = output_widget
        self.progress_bar = progress_bar
        self.contents = ""

    def write(self, s):
        self.contents += s
        # self.output_widget.text(self.contents)

        # Simple epoch % progress extractor
        if "epoch" in s and self.progress_bar:
            import re
            match = re.search(r"'epoch':\s*([\d\.]+)", s)
            if match:
                try:
                    epoch_val = float(match.group(1))
                    percent = min(int(epoch_val * 100), 100)
                    self.progress_bar.progress(percent)
                except:
                    pass

    def flush(self):
        pass
