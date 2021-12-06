use ndarray::prelude::*;

const WEIGHT_DIV: f64 = 16384f64;

pub fn pretty_genome(genome: &Vec<u32>) -> String {
    genome.iter().map(|g| format!("-{:08x}", g)).collect::<String>()
}

pub fn mutate_gene(gene: u32, n_bits: usize) -> u32 {
    use rand::prelude::*;
    let mut changes = 0;
    for _ in 0..n_bits {
        let pos: u32 = random();
        changes |= 1 << (pos % 32);
    }
    return gene ^ changes;
}

#[derive(Debug)]
enum SrcNeuron {
    Input(usize),
    Internal(usize),
}

#[derive(Debug)]
enum DstNeuron {
    Output(usize),
    Internal(usize),
}

#[derive(Debug)]
pub struct Synapse {
    neuron_src: SrcNeuron,
    neuron_dst: DstNeuron,
    weight: f64,
}

impl Synapse {
    pub fn from_gene(gene: u32) -> Synapse {
        let src = gene >> 24;
        let dst = (gene >> 16) & 0xff;
        let w = ((gene) & 0xffff) as i16;

        Synapse {
            neuron_src: if (src & 0x80) == 0x80 {
                SrcNeuron::Internal(src as usize & 0x7f)
            } else {
                SrcNeuron::Input(src as usize & 0x7f)
            },
            neuron_dst: if (dst & 0x80) == 0x80 {
                DstNeuron::Internal(dst as usize & 0x7f)
            } else {
                DstNeuron::Output(dst as usize & 0x7f)
            },
            weight: (w as f64) / WEIGHT_DIV,
        }
    }

    pub fn clip(&self, input_neurons: usize, internal_neurons: usize, output_neurons: usize) -> Synapse {
        Synapse {
            neuron_src: match self.neuron_src {
                SrcNeuron::Input(x) => SrcNeuron::Input(x % input_neurons),
                SrcNeuron::Internal(x) => if internal_neurons > 0 {
                    SrcNeuron::Internal(x % internal_neurons)
                } else {
                    SrcNeuron::Input(x % input_neurons)
                },
            },
            neuron_dst: match self.neuron_dst {
                DstNeuron::Output(x) => DstNeuron::Output(x % output_neurons),
                DstNeuron::Internal(x) => if internal_neurons > 0 {
                    DstNeuron::Internal(x % internal_neurons)
                } else {
                    DstNeuron::Output(x % output_neurons)
                },
            },
            weight: self.weight,
        }
    }

    pub fn to_gene(&self) -> u32 {
        let src = match self.neuron_src {
            SrcNeuron::Input(index) => index,
            SrcNeuron::Internal(index) => 0x80 | index,
        } as u32;
        let dst = match self.neuron_dst {
            DstNeuron::Output(index) => index,
            DstNeuron::Internal(index) => 0x80 | index,
        } as u32;
        let w = (self.weight * WEIGHT_DIV) as i16 as u16;
        return (src << 24) | (dst << 16) | (w as u32);
    }

    pub fn to_internal_neuron(&self) -> bool {
        match self.neuron_dst {
            DstNeuron::Internal(_) => true,
            _ => false,
        }
    }

    pub fn activate(&self, inputs: &Array<f64,Ix1>, internals_in: &Array<f64,Ix1>, internal_outs: &mut Array<f64,Ix1>, outputs: &mut Array<f64,Ix1>) {
        let (src, ridx) = match self.neuron_src {
            SrcNeuron::Input(index) => (inputs, index),
            SrcNeuron::Internal(index) => (internals_in, index),
        };
        if ridx >= src.len() {
            return;
        }

        let (dst, widx) = match self.neuron_dst {
            DstNeuron::Output(index) => (outputs, index),
            DstNeuron::Internal(index) => (internal_outs, index),
        };
        //println!("{:?}({:?}) -> {:?}", self, src[ridx], self.weight * src[ridx]);
        if widx < dst.len() {
            dst[widx] += self.weight * src[ridx];
        }
    }
}
