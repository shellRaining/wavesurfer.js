export default class FFT {
  private bufferSize: number
  private sinTable: Float32Array
  private cosTable: Float32Array
  private windowValues: Float32Array
  private reverseTable: Uint32Array
  private peak: number

  constructor(bufferSize: number, sampleRate: number, windowFunc: string, alpha?: number) {
    this.bufferSize = bufferSize

    this.sinTable = new Float32Array(bufferSize)
    this.cosTable = new Float32Array(bufferSize)
    this.windowValues = new Float32Array(bufferSize)
    this.reverseTable = new Uint32Array(bufferSize)

    this.peak = 0

    this.initWindowFunction(windowFunc, alpha)
    this.initReverseBits()
    this.initTrigTables()
  }

  private initWindowFunction(windowFunc: string, alpha?: number): void {
    let i: number
    switch (windowFunc) {
      case 'bartlett':
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] =
            (2 / (this.bufferSize - 1)) * ((this.bufferSize - 1) / 2 - Math.abs(i - (this.bufferSize - 1) / 2))
        }
        break
      case 'bartlettHann':
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] =
            0.62 -
            0.48 * Math.abs(i / (this.bufferSize - 1) - 0.5) -
            0.38 * Math.cos((Math.PI * 2 * i) / (this.bufferSize - 1))
        }
        break
      case 'blackman':
        alpha = alpha || 0.16
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] =
            (1 - alpha) / 2 -
            0.5 * Math.cos((Math.PI * 2 * i) / (this.bufferSize - 1)) +
            (alpha / 2) * Math.cos((4 * Math.PI * i) / (this.bufferSize - 1))
        }
        break
      case 'cosine':
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] = Math.cos((Math.PI * i) / (this.bufferSize - 1) - Math.PI / 2)
        }
        break
      case 'gauss':
        alpha = alpha || 0.25
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] = Math.pow(
            Math.E,
            -0.5 * Math.pow((i - (this.bufferSize - 1) / 2) / ((alpha * (this.bufferSize - 1)) / 2), 2),
          )
        }
        break
      case 'hamming':
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] = 0.54 - 0.46 * Math.cos((Math.PI * 2 * i) / (this.bufferSize - 1))
        }
        break
      case 'hann':
      case undefined:
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] = 0.5 * (1 - Math.cos((Math.PI * 2 * i) / (this.bufferSize - 1)))
        }
        break
      case 'lanczoz':
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] =
            Math.sin(Math.PI * ((2 * i) / (this.bufferSize - 1) - 1)) /
            (Math.PI * ((2 * i) / (this.bufferSize - 1) - 1))
        }
        break
      case 'rectangular':
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] = 1
        }
        break
      case 'triangular':
        for (i = 0; i < this.bufferSize; i++) {
          this.windowValues[i] = (2 / this.bufferSize) * (this.bufferSize / 2 - Math.abs(i - (this.bufferSize - 1) / 2))
        }
        break
      default:
        throw new Error(`No such window function '${windowFunc}'`)
    }
  }

  private initReverseBits(): void {
    let limit = 1
    let bit = this.bufferSize >> 1

    while (limit < this.bufferSize) {
      for (let i = 0; i < limit; i++) {
        this.reverseTable[i + limit] = this.reverseTable[i] + bit
      }

      limit = limit << 1
      bit = bit >> 1
    }
  }

  private initTrigTables(): void {
    for (let i = 0; i < this.bufferSize; i++) {
      this.sinTable[i] = Math.sin(-Math.PI / i)
      this.cosTable[i] = Math.cos(-Math.PI / i)
    }
  }

  public calculateSpectrum(buffer: Float32Array): Float32Array {
    const bufferSize = this.bufferSize
    const { cosTable, sinTable, reverseTable, windowValues } = this
    const real = new Float32Array(bufferSize)
    const imag = new Float32Array(bufferSize)
    const bSi = 2 / bufferSize
    const spectrum = new Float32Array(bufferSize >> 1)

    // Input data windowing and reordering
    for (let i = 0; i < bufferSize; i++) {
      real[i] = buffer[reverseTable[i]] * windowValues[reverseTable[i]]
    }

    // Main FFT loop
    for (let halfSize = 1; halfSize < bufferSize; halfSize <<= 1) {
      const phaseShiftStepReal = cosTable[halfSize]
      const phaseShiftStepImag = sinTable[halfSize]
      let currentPhaseShiftReal = 1
      let currentPhaseShiftImag = 0

      for (let fftStep = 0; fftStep < halfSize; fftStep++) {
        for (let i = fftStep; i < bufferSize; i += halfSize << 1) {
          const off = i + halfSize
          const tr = currentPhaseShiftReal * real[off] - currentPhaseShiftImag * imag[off]
          const ti = currentPhaseShiftReal * imag[off] + currentPhaseShiftImag * real[off]

          real[off] = real[i] - tr
          imag[off] = imag[i] - ti
          real[i] += tr
          imag[i] += ti
        }

        const tmpReal = currentPhaseShiftReal
        currentPhaseShiftReal = tmpReal * phaseShiftStepReal - currentPhaseShiftImag * phaseShiftStepImag
        currentPhaseShiftImag = tmpReal * phaseShiftStepImag + currentPhaseShiftImag * phaseShiftStepReal
      }
    }

    // Compute magnitude spectrum
    const N = bufferSize >> 1
    for (let i = 0; i < N; i++) {
      const mag = bSi * Math.hypot(real[i], imag[i])
      if (mag > this.peak) {
        this.peak = mag
      }
      spectrum[i] = mag
    }

    return spectrum
  }
}
