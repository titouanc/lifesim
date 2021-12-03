struct Query<K> {x: K, y: K, w: K, h: K}

struct Node<K, V> {
    top_left: Box<Option<Node<K, V>>>,
    top_right: Box<Option<Node<K, V>>>,
    bottom_left: Box<Option<Node<K, V>>>,
    bottom_right: Box<Option<Node<K, V>>>,
    value: V,
    x: K,
    y: K,
}

impl<K: std::cmp::PartialOrd + std::ops::Add<Output = K> + Copy, V> Node<K, V> {
    fn new(x: K, y: K, value: V) -> Node<K, V> {
        Node {
            top_left: Box::new(None),
            top_right: Box::new(None),
            bottom_left: Box::new(None),
            bottom_right: Box::new(None),
            value: value,
            x: x,
            y: y
        }
    }

    fn insert(&mut self, x: K, y: K, value: V) {
        let quadrant = if x < self.x {
            if y < self.y {
                &mut *self.bottom_left
            } else {
                &mut *self.top_left
            }
        } else {
            if y < self.y {
                &mut *self.bottom_right
            } else {
                &mut *self.top_right
            }
        };
        match quadrant {
            Some(node) => node.insert(x, y, value),
            None => *quadrant = Some(Node::new(x, y, value)),
        }
    }

    fn query<'a>(&'a self, query: & Query<K>, into: &mut Vec<&'a V>) {
        let i_am_in_the_query_rectangle =
            query.x < self.x
            && self.x < query.x + query.w
            && query.y < self.y
            && self.y < query.y + query.h;

        if i_am_in_the_query_rectangle {
            into.push(&self.value);
        }
        if query.x < self.x {
            if query.y < self.y {
                match &*self.bottom_left {
                    Some(node) => node.query(query, into),
                    _ => {},
                }
            }
            if query.y + query.h > self.y {
                match &*self.top_left {
                    Some(node) => node.query(query, into),
                    _ => {},
                }   
            }
        }
        if query.x + query.w > self.x {
            if query.y < self.y {
                match &*self.bottom_right {
                    Some(node) => node.query(query, into),
                    _ => {},
                }
            }
            if query.y + query.h > self.y {
                match &*self.top_right {
                    Some(node) => node.query(query, into),
                    _ => {},
                }   
            }   
        }
    }
}

pub struct QuadTree<K, V> {
    root: Option<Node<K, V>>,
}

impl<K: std::cmp::PartialOrd + std::ops::Add<Output = K> + Copy, V> QuadTree<K, V> {
    pub fn new() -> QuadTree<K, V> {
        QuadTree {root: None}
    }

    pub fn insert(&mut self, x: K, y: K, value: V) {
        match &mut self.root {
            Some(node) => node.insert(x, y, value),
            None => self.root = Some(Node::new(x, y, value)),
        }
    }

    pub fn query_rect<'a>(&'a self, x: K, y: K, w: K, h: K) -> Vec<&'a V> {
        let mut res = vec![];
        let q = Query {x: x, y: y, w: w, h: h};
        match &self.root {
            Some(node) => node.query(&q, &mut res),
            _ => {},
        }
        return res;
    }
}

/*
fn main() {
    use rand::prelude::*;

    let mut q: QuadTree<f64, i32> = QuadTree::new();

    for i in 0..1000000 {
        let x: f64 = random();
        let y: f64 = random();
        q.insert(2.*x-1., 2.*y-1., i);
    }

    let res = q.query_rect(0.0, 0.0, 1.0, 1.0);
    println!("Result: {}", res.len());
}
*/