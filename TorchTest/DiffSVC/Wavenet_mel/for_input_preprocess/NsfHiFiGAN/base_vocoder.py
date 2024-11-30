VOCODERS = {}


def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


class BaseVocoder:
    def spec2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError

    @staticmethod
    def wav2spec(wav_fn, hparams):
        """

        :param hparams:
        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        raise NotImplementedError
