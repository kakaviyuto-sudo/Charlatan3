class NeuroError(Exception):
    pass

class DependencyMissingError(NeuroError):
    pass

class ModelUnavailableError(NeuroError):
    pass

class TTSUnavailableError(NeuroError):
    pass

class VTubeConnectionError(NeuroError):
    pass

class BrowserFetchError(NeuroError):
    pass

class TranscriptArchiveError(NeuroError):
    pass