mod synapse;
use synapse::*;

extern crate simple;

use std::collections::HashSet;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;

pub fn sigmoid(x: f64) -> f64 {
    2f64 / (1f64 + f64::exp(-x)) - 1f64
}

pub fn distance(x1: &Array<f64, Ix1>, x2: &Array<f64, Ix1>) -> f64 {
    let dx = x1 - x2;
    return f64::sqrt((&dx*&dx).sum())
}

pub trait Area {
    fn area(&self) -> f64;
}

#[derive(Debug)]
pub struct WorldSense {
    // World coordinates
    coord_top_left: Array<f64, Ix1>,
    coord_bottom_right: Array<f64, Ix1>,

    // Age of the world, in ticks
    age: usize,

    // Number of creatures in vicinity
    top_left: usize,
    top_right: usize,
    bottom_left: usize,
    bottom_right: usize,
    top_left_blocked: usize,
    top_right_blocked: usize,
    bottom_left_blocked: usize,
    bottom_right_blocked: usize,
}

#[derive(Debug)]
pub struct Creature {
    // "Biological structure"
    adn: Vec<u32>,
    synapses: Vec<Synapse>,
    internal_neurons: Array<f64, Ix1>,
    size: f64,
    // State in the world
    position: Array<f64, Ix1>,
    velocity: Array<f64, Ix1>,
    blocked: bool,
    // stats
    moves: usize,
}

impl Area for Creature {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.size * self.size
    }
}

const SENSE_RANGE_MULT: f64 = 5.0;
const N_INPUT_NEURONS: usize = 15;
const N_OUTPUT_NEURONS: usize = 2;

impl Creature {
    pub fn random(n_genes: usize) -> Creature {
        use rand::prelude::*;

        let mut genome: Vec<u32> = vec![random()];
        for _ in 0..n_genes {
            genome.push(random());
        }
        Creature::from_genome(&genome)
    }

    pub fn from_genome(genome: &Vec<u32>) -> Creature {
        if genome.len() < 1 {
            panic!("Invalid genome");
        }

        let characteristics = genome[0];
        let size = 3.0 + (characteristics & 0xff) as f64 / 8.0;
        let n_internal_neurons = 1 + ((characteristics >> 8) & 0x01) as usize;

        Creature {
            adn: genome.iter().skip(1).map(|x| *x).collect(),
            position: array![0.0, 0.0],
            velocity: array![0.0, 0.0],
            size: size,
            synapses: genome.iter().skip(1).map(|x|Synapse::from_gene(*x).clip(N_INPUT_NEURONS, n_internal_neurons, N_OUTPUT_NEURONS)).collect(),
            internal_neurons: Array::zeros((n_internal_neurons,)),
            blocked: false,
            moves: 0,
        }
    }

    pub fn genome(&self) -> Vec<u32> {
        let size = (8.0 * (self.size - 3.0)) as u32;
        let characteristics = ((self.internal_neurons.len() - 1) << 8) as u32 | size;
        let mut res = vec![characteristics];
        res.extend(&self.adn);
        return res;
    }

    pub fn reproduce(&self, mutation_prob: f64) -> Creature {
        use rand::prelude::*;
        let genome = self.genome().iter().map(|gene| {
            let p: f64 = random();
            if p < mutation_prob {
                mutate_gene(*gene, 3)
            } else {
                *gene
            }
        }).collect();
        Creature::from_genome(&genome)
    }

    pub fn randomize_position(&mut self, bounds: &Array<f64, Ix1>) {
        use ndarray_rand::RandomExt;
        let pos = Array::random((2,), Uniform::new(0., 1.0)) * bounds;
        self.position.assign(&pos);
    }

    pub fn next_position(&self) -> Array<f64, Ix1> {
        &self.position + &self.velocity
    }

    pub fn think(&mut self, sense: &WorldSense) {
        let dpos = &sense.coord_bottom_right - &sense.coord_top_left;
        let pos = (&self.position - &sense.coord_top_left) / dpos - 0.5;
        let input_neurons = array![
            self.size,
            pos[0],
            pos[1],
            self.velocity[0],
            self.velocity[1],
            self.blocked as i64 as f64,
            self.moves as f64,
            sense.top_left as f64,
            sense.top_right as f64,
            sense.bottom_left as f64,
            sense.bottom_right as f64,
            sense.top_left_blocked as f64,
            sense.top_right_blocked as f64,
            sense.bottom_left_blocked as f64,
            sense.bottom_right_blocked as f64,
            // f64::sin(sense.age as f64 / 10.0),
            // f64::sin(sense.age as f64 / 100.0),
            // f64::sin(sense.age as f64 / 1000.0),
        ];
        assert!(input_neurons.len() == N_INPUT_NEURONS);
        let mut internal_neurons = Array::zeros(self.internal_neurons.raw_dim());
        let mut output_neurons = Array::zeros((N_OUTPUT_NEURONS,));

        // First pass: all synapses to internal neurons
        for syn in &self.synapses {
            if syn.to_internal_neuron() {
                syn.activate(&input_neurons, &self.internal_neurons, &mut internal_neurons, &mut output_neurons);
            }
        }
        self.internal_neurons.assign(&internal_neurons.mapv(sigmoid));

        // Second pass: all synapses to output neurons
        for syn in &self.synapses {
            if ! syn.to_internal_neuron() {
                syn.activate(&input_neurons, &self.internal_neurons, &mut internal_neurons, &mut output_neurons);
            }
        }
        output_neurons = output_neurons.mapv(sigmoid) * 5.0;

        // Get output neurons
        self.velocity.assign(&output_neurons.slice(s![..2]));
    }

    pub fn act(&mut self) {
        if ! self.blocked {
            self.position += &self.velocity;
            self.moves += 1;
        }
    }
}

pub trait Collide {
    fn collide(&self, creature: &Creature) -> bool;

    fn collide_next(&self, creature: &Creature) -> bool;
}

impl Collide for Creature {
    fn collide(&self, other: &Creature) -> bool {
        distance(&self.position, &other.position) < (self.size + other.size)
    }

    fn collide_next(&self, other: &Creature) -> bool {
        distance(&self.position, &other.next_position()) < (self.size + other.size)
    }
}

#[derive(Debug, Clone)]
pub struct Garden {
    top: f64,
    left: f64,
    bottom: f64,
    right: f64,
}

impl Collide for Garden {
    fn collide(&self, creature: &Creature) -> bool {
        self.top < creature.position[0] - creature.size &&
        creature.position[0] + creature.size < self.bottom &&
        self.left < creature.position[1] - creature.size &&
        creature.position[1] + creature.size < self.right
    }

    fn collide_next(&self, creature: &Creature) -> bool {
        false
    }
}

impl Area for Garden {
    fn area(&self) -> f64 {
        (self.bottom - self.top) * (self.right - self.left)
    }
}

pub struct World {
    top_left: Array<f64, Ix1>,
    bottom_right: Array<f64, Ix1>,
    population: Vec<Creature>,
    age: usize,
    gardens: Vec<Garden>,
}

impl Area for World {
    fn area(&self) -> f64 {
        let wh = &self.bottom_right - &self.top_left;
        wh[0] * wh[1]
    }
}

impl Collide for World {
    fn collide(&self, creature: &Creature) -> bool {
        let tl = &creature.position - &self.top_left;
        let br = &self.bottom_right - &creature.position;
        tl.iter().chain(br.iter()).any(|x| x < &creature.size)
    }

    fn collide_next(&self, creature: &Creature) -> bool {
        let pos = creature.next_position();
        let tl = &pos - &self.top_left;
        let br = &self.bottom_right - &pos;
        tl.iter().chain(br.iter()).any(|x| x < &creature.size)
    }
}

impl World {
    pub fn new(width: usize, height: usize, gardens: &Vec<Garden>) -> World {
        World {
            top_left: array![0., 0.],
            bottom_right: array![height as f64, width as f64],
            population: vec![],
            age: 0,
            gardens: gardens.to_vec(),
        }
    }

    pub fn can_add_creature(&self, creature: &Creature) -> bool {
        ! (
            self.collide(creature) ||
            self.population.iter().any(|c| c.collide(creature))
        )
    }

    pub fn add_creature(&mut self, creature: Creature) {
        assert!(self.can_add_creature(&creature));
        self.population.push(creature);
    }

    pub fn sense(&self, w: f64, h: f64, s: f64) -> WorldSense {
        let mut res = WorldSense {
            coord_top_left: self.top_left.clone(),
            coord_bottom_right: self.bottom_right.clone(),
            top_left: 0,
            top_right: 0,
            bottom_left: 0,
            bottom_right: 0,
            top_left_blocked: 0,
            top_right_blocked: 0,
            bottom_left_blocked: 0,
            bottom_right_blocked: 0,
            age: self.age,
        };

        let p = array![h, w];
        for c in &self.population {
            if distance(&p, &c.position) < s {
                if c.position[0] < p[0] {
                    if c.position[1] < p[1] {
                        res.top_left += 1;
                        if c.blocked {
                            res.top_left_blocked += 1;
                        }
                    } else {
                        res.top_right += 1;
                        if c.blocked {
                            res.top_right_blocked += 1;
                        }
                    }
                } else {
                    if c.position[1] < p[1] {
                        res.bottom_left += 1;
                        if c.blocked {
                            res.bottom_left_blocked += 1;
                        }
                    } else {
                        res.bottom_right += 1;
                        if c.blocked {
                            res.bottom_right_blocked += 1;
                        }
                    }
                }
            }
        }

        return res;
    }

    pub fn live(&mut self) {
        self.age += 1;

        let sense: Vec<WorldSense> =
            self.population
                .par_iter()
                .map(|c| self.sense(c.position[0], c.position[1], SENSE_RANGE_MULT * c.size))
                .collect();

        // Let the creatures think
        for (creature, s) in &mut self.population.iter_mut().zip(sense) {
            creature.think(&s);
        }

        // Check for collisions
        let collisions: Vec<bool> =
            self.population
                .par_iter()
                .enumerate()
                .map(|(i, creature)|
                    self.collide_next(creature)
                    || self.population
                           .iter()
                           .enumerate()
                           .any(|(j, c)| i != j && c.collide_next(creature))
                )
                .collect();

        for (creature, blocked) in &mut self.population.iter_mut().zip(collisions) {
            creature.blocked = blocked;
            creature.act();
        }
    }

    pub fn size(&self) -> Array<f64, Ix1> {
        return &self.bottom_right - &self.top_left;
    }
}

const N_CREATURES: u32 = 100;
const N_GENES: u32 = 16;

const WORLD_WIDTH: u32 = 1920;
const WORLD_HEIGHT: u32 = 1080;
const MUTATION_P: f64 = 3.0 / (N_CREATURES * N_GENES) as f64;

fn simulate(genomes: &Vec<Vec<u32>>, gardens: &Vec<Garden>) -> Vec<Vec<u32>> {
    let mut world = World::new(WORLD_WIDTH as usize, WORLD_HEIGHT as usize, gardens);
    for genome in genomes {
        let mut creature = Creature::from_genome(genome);
        while ! world.can_add_creature(&creature){
            creature.randomize_position(&world.bottom_right);
        }
        world.add_creature(creature);
    }

    let mut app = simple::Window::new("Creatures", WORLD_WIDTH as u16, WORLD_HEIGHT as u16);
    for _ in 0..300 {
        if ! app.next_frame(){
            panic!("STOP");
        }
        world.live();

        app.clear();
        app.set_color(0, 120, 0, 255);
        for garden in gardens {
            app.fill_rect(simple::Rect::new(
                garden.left as i32,
                garden.top as i32,
                (garden.right - garden.left) as u32,
                (garden.bottom - garden.top) as u32
            ));
        }

        for creature in &world.population {
            let tl = &creature.position - creature.size;

            let in_garden = gardens.iter().any(|garden| garden.collide(creature));

            if in_garden {
                app.set_color(120, 255, 120, 255)
            } else if creature.blocked {
                app.set_color(255, 120, 120, 255);
            } else {
                app.set_color(255, 255, 120, 255);
            }

            app.fill_rect(simple::Rect::new(
                tl[1] as i32, 
                tl[0] as i32,
                2 * creature.size as u32,
                2 * creature.size as u32
            ));
        }
    }

    world.population
         .iter()
         .filter(|creature| gardens.iter().any(|garden| garden.collide(creature)))
         .map(|creature| creature.genome())
         .collect()
}

fn main() {
    let mut run = 1;
    let mut generation = 0;
    let mut genomes: Vec<Vec<u32>> = (0..N_CREATURES).map(|_| Creature::random(N_GENES as usize).genome()).collect();

    let gardens = vec![
        // Garden {
        //     top: (2*WORLD_HEIGHT/7) as f64,
        //     left: (2*WORLD_WIDTH/7) as f64,
        //     bottom: (5*WORLD_HEIGHT/7) as f64,
        //     right: (5*WORLD_WIDTH/7) as f64,
        // },
        Garden {
            top: (WORLD_HEIGHT/7) as f64,
            left: (0) as f64,
            bottom: (6*WORLD_HEIGHT/7) as f64,
            right: (WORLD_WIDTH/12) as f64,
        },
        Garden {
            top: (WORLD_HEIGHT/7) as f64,
            left: (11*WORLD_WIDTH/12) as f64,
            bottom: (6*WORLD_HEIGHT/7) as f64,
            right: (WORLD_WIDTH) as f64,
        },
    ];

    let gardens_area: f64 = gardens.iter().map(Garden::area).sum();
    let gardens_ratio = gardens_area / (WORLD_WIDTH * WORLD_HEIGHT) as f64;

    loop {
        let genomes_distinct: HashSet<String> = genomes.iter().map(pretty_genome).collect();
        let diversity_ratio = (genomes_distinct.len() as f64) / (N_CREATURES as f64);

        let survivors = simulate(&genomes, &gardens);

        let survivors_ratio = (survivors.len() as f64) / (N_CREATURES as f64);
        let score = if survivors_ratio < gardens_ratio {
            survivors_ratio / gardens_ratio - 1.
        } else {
            (survivors_ratio - gardens_ratio) / (1. - gardens_ratio)
        };
        let pretty_score = f64::round(100. * score) as i32;
        println!("Civilisation {} / Generation: {} / Diversity: {}% / Survivors: {}% => Score: \x1b[1m{}\x1b[0m", run, generation, (100. * diversity_ratio) as i32, (100. * survivors_ratio) as i32, pretty_score);
        
        if survivors.len() == 0 {
            generation = 0;
            run += 1;
            genomes = (0..N_CREATURES).map(|_| Creature::random(N_GENES as usize).genome()).collect();
        } else {
            genomes.clear();

            while genomes.len() < N_CREATURES as usize {
                let parent = &survivors[genomes.len() % survivors.len()];
                genomes.push(Creature::from_genome(parent).reproduce(MUTATION_P).genome());
            }
            generation += 1;
        }
    }
}
