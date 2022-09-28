use arrayvec::ArrayVec;
use std::collections::BTreeMap;
use wasm_bindgen::prelude::*;

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct SsaVariable(usize);
impl SsaVariable {
    pub const INVALID: Self = Self(usize::MAX);
}

#[derive(Debug)]
enum Op {
    Binary {
        name: String,
        out: SsaVariable,
        lhs: SsaVariable,
        rhs: SsaVariable,
    },
    Unary {
        name: String,
        out: SsaVariable,
        lhs: SsaVariable,
    },
    Nonary {
        name: String,
        out: SsaVariable,
    },
    Output {
        out: SsaVariable,
    },
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Register(usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Memory(usize);

#[derive(Debug)]
enum AsmOp {
    Load {
        src: Memory,
        dst: Register,
    },
    Store {
        src: Register,
        dst: Memory,
    },
    Binary {
        name: String,
        out: Register,
        lhs: Register,
        rhs: Register,
    },
    Unary {
        name: String,
        out: Register,
        lhs: Register,
    },
    Nonary {
        name: String,
        out: Register,
    },
    Output {
        out: Register,
    },
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Allocation {
    Register(Register),
    Memory(Memory),
    Unassigned,
}

/// Cheap and cheerful single-pass register allocation
pub struct RegisterAllocator<const N: usize> {
    /// Map from a `SsaVariable` in the original tape to the relevant allocation
    allocations: Vec<Allocation>,

    /// Map from a particular register to the index in the original tape that's
    /// using that register, or `SsaVariable::INVALID` if the register is currently
    /// unused.
    registers: [SsaVariable; N],

    /// Stores a least-recently-used list of register
    register_lru: Lru<N>,

    /// Available registers (index < N)
    spare_registers: ArrayVec<Register, N>,

    /// Available extended registers (index >= N)
    spare_memory: Vec<Memory>,

    /// Total allocated slots (registers and memory)
    total_slots: usize,

    /// Output slots, assembled in reverse order
    out: Vec<AsmOp>,
}

impl<const N: usize> RegisterAllocator<N> {
    /// Builds a new `RegisterAllocator`.
    ///
    /// Upon construction, SSA register 0 is bound to local register 0; you
    /// would be well advised to use it as the output of your function.
    pub fn new(size: usize) -> Self {
        let mut out = Self {
            allocations: vec![],

            registers: [SsaVariable(usize::MAX); N],
            register_lru: Lru::new(),

            spare_registers: ArrayVec::new(),
            spare_memory: Vec::with_capacity(1024),

            total_slots: 1,
            out: Vec::with_capacity(1024),
        };
        out.bind_register(SsaVariable(0), Register(0));
        out
    }

    fn get_allocation(&mut self, i: SsaVariable) -> &mut Allocation {
        if i.0 >= self.allocations.len() {
            self.allocations.resize(i.0 + 1, Allocation::Unassigned);
        }
        &mut self.allocations[i.0]
    }

    /// Returns an available memory slot.
    ///
    /// Memory is treated as unlimited; if we don't have any spare slots, then
    /// we'll assign a new one (incrementing `self.total_slots`).
    ///
    /// > If there's one thing I love  
    /// > It's an infinite resource  
    /// > If there's one thing worth loving  
    /// > It's a surplus of supplies
    fn get_memory(&mut self) -> Memory {
        if let Some(p) = self.spare_memory.pop() {
            p
        } else {
            let out = self.total_slots;
            self.total_slots += 1;
            Memory(out)
        }
    }

    /// Finds the oldest register
    ///
    /// This is useful when deciding which register to evict to make room
    fn oldest_reg(&mut self) -> Register {
        Register(self.register_lru.pop())
    }

    /// Return an unoccupied register, if available
    fn get_spare_register(&mut self) -> Option<Register> {
        self.spare_registers.pop().or_else(|| {
            if self.total_slots < N {
                let reg = self.total_slots;
                assert!(self.registers[reg as usize] == SsaVariable::INVALID);
                self.total_slots += 1;
                Some(Register(reg))
            } else {
                None
            }
        })
    }

    /// Get a register by any means necessary, including evicting an existing
    /// binding.
    fn get_register(&mut self) -> Register {
        if let Some(reg) = self.get_spare_register() {
            assert_eq!(self.registers[reg.0], SsaVariable::INVALID);
            self.register_lru.poke(reg.0);
            reg
        } else {
            // Slot is in memory, and no spare register is available
            let reg = self.oldest_reg();

            // Here's where it will go:
            let mem = self.get_memory();

            // Whoever was previously using you is in for a surprise
            let prev_node = self.registers[reg.0];
            *self.get_allocation(prev_node) = Allocation::Memory(mem);

            // This register is now unassigned
            self.registers[reg.0] = SsaVariable::INVALID;

            self.out.push(AsmOp::Load { dst: reg, src: mem });
            reg
        }
    }

    /// Rebind the given register to a new `SsaVariable`, which must be in memory
    ///
    /// The target register must already be bound; it is unassigned by this
    /// function call.
    fn rebind_register(&mut self, n: SsaVariable, reg: Register) {
        assert!(matches!(*self.get_allocation(n), Allocation::Memory(..)));

        let prev_node = self.registers[reg.0];
        assert!(prev_node != SsaVariable::INVALID);
        *self.get_allocation(prev_node) = Allocation::Unassigned;

        // Bind the register and update its use time
        self.registers[reg.0] = n;
        *self.get_allocation(n) = Allocation::Register(reg);
        self.register_lru.poke(reg.0);
    }

    /// Bind the given register to a new `SsaVariable`, which must be in memory
    ///
    /// The target register must be unbound
    fn bind_register(&mut self, n: SsaVariable, reg: Register) {
        assert!(matches!(*self.get_allocation(n), Allocation::Memory(..)));
        assert!(self.registers[reg.0] == SsaVariable::INVALID);

        // Bind the register and update its use time
        self.registers[reg.0] = n;
        *self.get_allocation(n) = Allocation::Register(reg);
        self.register_lru.poke(reg.0);
    }

    /// Release a register back to the pool of spares
    fn release_reg(&mut self, reg: Register) {
        // Release the output register, so it could be used for inputs
        assert!(reg.0 < N);

        let node = self.registers[reg.0];
        assert!(node != SsaVariable::INVALID);

        self.registers[reg.0] = SsaVariable::INVALID;
        self.spare_registers.push(reg);

        // Modifying self.allocations isn't strictly necessary, but could help
        // us detect logical errors (since it should never be used after this)
        *self.get_allocation(node) = Allocation::Unassigned;
    }

    fn release_mem(&mut self, mem: Memory) {
        assert!(mem.0 >= N);
        self.spare_memory.push(mem);
        // This leaves self.allocations[...] still pointing to the memory slot,
        // but that's okay, because it should never be used (and we have no way
        // of finding it)
    }

    /// Returns a register that is bound to the given SSA variable
    ///
    /// If the given SSA variable is not already bound to a register, then we
    /// evict the oldest register using `Self::get_register`, with the
    /// appropriate set of LOAD/STORE operations.
    fn get_out_reg(&mut self, out: SsaVariable) -> Register {
        use Allocation::*;
        match *self.get_allocation(out) {
            Register(r_x) => r_x,
            Memory(m_x) => {
                // TODO: this could be more efficient with a Swap instruction,
                // since we know that we're about to free a memory slot.
                let r_a = self.get_register();

                self.out.push(AsmOp::Store { src: r_a, dst: m_x });
                self.release_mem(m_x);
                self.bind_register(out, r_a);
                r_a
            }
            Unassigned => panic!("Cannot have unassigned output"),
        }
    }

    /// Lowers an operation that uses a single register into an
    /// [`AsmOp`](crate::asm::AsmOp), pushing it to the internal tape.
    ///
    /// This may also push `Load` or `Store` instructions to the internal tape,
    /// if there aren't enough spare registers.
    fn op_reg(&mut self, out: SsaVariable, arg: SsaVariable, name: &str) {
        let op = |out, lhs| AsmOp::Unary {
            out,
            lhs,
            name: name.to_owned(),
        };
        use Allocation::*;
        let r_x = self.get_out_reg(out);
        match *self.get_allocation(arg) {
            Register(r_y) => {
                assert!(r_x != r_y);
                self.out.push(op(r_x, r_y));
                self.release_reg(r_x);
            }
            Memory(m_y) => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);

                self.out.push(AsmOp::Store { src: r_x, dst: m_y });
                self.release_mem(m_y);
            }
            Unassigned => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);
            }
        }
    }

    /// Lowers a two-register operation into an [`AsmOp`](crate::asm::AsmOp),
    /// pushing it to the internal tape.
    ///
    /// Inputs are SSA registers from a [`Tape`](crate::tape::Tape), i.e.
    /// globally addressed.
    ///
    /// If there aren't enough spare registers, this may also push `Load` or
    /// `Store` instructions to the internal tape.  It's trickier than it
    /// sounds; look at the source code for a table showing all 18 (!) possible
    /// configurations.
    fn op_reg_reg(&mut self, out: SsaVariable, lhs: SsaVariable, rhs: SsaVariable, name: &str) {
        let op = |out, lhs, rhs| AsmOp::Binary {
            out,
            lhs,
            rhs,
            name: name.to_owned(),
        };
        use Allocation::*;
        let r_x = self.get_out_reg(out);
        match (*self.get_allocation(lhs), *self.get_allocation(rhs)) {
            (Register(r_y), Register(r_z)) => {
                self.out.push(op(r_x, r_y, r_z));
                self.release_reg(r_x);
            }
            (Memory(m_y), Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);

                self.out.push(AsmOp::Store { src: r_x, dst: m_y });
                self.release_mem(m_y);
            }
            (Register(r_y), Memory(m_z)) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);

                self.out.push(AsmOp::Store { src: r_x, dst: m_z });
                self.release_mem(m_z);
            }
            (Memory(m_y), Memory(m_z)) => {
                let r_a = if lhs == rhs { r_x } else { self.get_register() };

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_register(rhs, r_a);
                }

                self.out.push(AsmOp::Store { src: r_x, dst: m_y });
                self.release_mem(m_y);

                if lhs != rhs {
                    self.out.push(AsmOp::Store { src: r_a, dst: m_z });
                    self.release_mem(m_z);
                }
            }
            (Unassigned, Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);
            }
            (Register(r_y), Unassigned) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);
            }
            (Unassigned, Unassigned) => {
                let r_a = if lhs == rhs { r_x } else { self.get_register() };

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_register(rhs, r_a);
                }
            }
            (Unassigned, Memory(m_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_x, r_a));
                self.rebind_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_register(rhs, r_a);
                }

                self.out.push(AsmOp::Store { src: r_a, dst: m_z });
                self.release_mem(m_z);
            }
            (Memory(m_y), Unassigned) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_a, r_x));
                self.bind_register(lhs, r_a);
                if lhs != rhs {
                    self.rebind_register(rhs, r_x);
                }

                self.out.push(AsmOp::Store { src: r_a, dst: m_y });
                self.release_mem(m_y);
            }
        }
    }

    fn op_out_only(&mut self, out: SsaVariable, name: &str) {
        let r_x = self.get_out_reg(out);
        self.out.push(AsmOp::Nonary {
            out: r_x,
            name: name.to_owned(),
        });
        self.release_reg(r_x);
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Single node in the doubly-linked list
#[derive(Copy, Clone)]
struct LruNode {
    prev: usize,
    next: usize,
}

/// Dead-simple LRU cache, implemented as a doubly-linked list with a static
/// backing array.
///
/// ```text
///              <-- prev next -->
///      -------      -------      -------      --------
///      |  a  | <--> |  b  | <--> |  c  | <--> | head | <--|
///      -------      -------      -------      --------    |
///         ^                       oldest                  |
///         |-----------------------------------------------|
/// ```
pub struct Lru<const N: usize> {
    data: [LruNode; N],
    head: LruNode,
}

impl<const N: usize> Lru<N> {
    pub fn new() -> Self {
        let mut out = Self {
            head: LruNode {
                prev: N - 1,
                next: 0,
            },
            data: [LruNode { prev: 0, next: 0 }; N],
        };
        for i in 0..N {
            out.data[i].prev = i.checked_sub(1).unwrap_or(N);
            out.data[i].next = i + 1;
        }

        out
    }

    fn get_mut(&mut self, i: usize) -> &mut LruNode {
        assert!(i <= N);
        self.data.get_mut(i).unwrap_or(&mut self.head)
    }

    /// Mark the given node as newest
    pub fn poke(&mut self, i: usize) {
        let prev_newest = self.head.next;
        if prev_newest != i {
            // Remove this node from the list
            self.get_mut(self.data[i].prev).next = self.data[i].next;
            self.get_mut(self.data[i].next).prev = self.data[i].prev;

            // Reinsert the node between prev_newest and the head
            self.data[prev_newest].prev = i;
            self.head.next = i;
            self.data[i].next = prev_newest;
            self.data[i].prev = N;
        }
    }
    /// Look up the oldest node in the list, marking it as newest
    pub fn pop(&mut self) -> usize {
        let out = self.head.prev;
        self.poke(out);
        out
    }
}

////////////////////////////////////////////////////////////////////////////////

fn main() {
    println!("Hello, world!");
}

#[wasm_bindgen]
pub fn do_the_thing(s: String) -> String {
    format!("{:?}", parse(&s))
}

fn parse(s: &str) -> Result<Vec<Op>, String> {
    let mut vars = BTreeMap::new();
    let mut out = vec![];
    for line in s.split('\n') {
        out.push(parse_line(line, &mut vars)?);
    }
    Ok(out)
}

fn parse_line(mut line: &str, vars: &mut BTreeMap<String, SsaVariable>) -> Result<Op, String> {
    let line = line.trim();
    if let Some(rest) = line.strip_prefix("OUTPUT(") {
        if let Some(name) = rest.strip_suffix(')') {
            if let Some(out) = vars.get(name) {
                return Ok(Op::Output { out: *out });
            } else {
                return Err(format!("Unknown output: {name}"));
            }
        } else {
            return Err("Missing close parenthesis".to_owned());
        }
    }

    let mut iter = line.split('=');

    let out = iter.next().ok_or_else(|| "Missing =".to_owned())?.trim();
    if vars.contains_key(out) {
        return Err(format!("Duplicated assignment: {out}"));
    }
    let ssa_out = SsaVariable(vars.len());
    vars.insert(out.to_owned(), ssa_out);

    let op = iter
        .next()
        .ok_or_else(|| "Missing operation".to_owned())?
        .trim();
    let (op, rest) = op
        .split_once('(')
        .ok_or_else(|| "Missing opening (".to_owned())?;
    let rest = rest
        .strip_suffix(')')
        .ok_or_else(|| "Missing closing )".to_owned())?;

    let mut ssa_vars = vec![];
    for v in rest
        .trim()
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
    {
        match vars.get(v) {
            Some(s) => ssa_vars.push(*s),
            None => return Err(format!("Unknown argument: '{v}'")),
        }
    }
    match ssa_vars.len() {
        0 => Ok(Op::Nonary {
            name: op.to_string(),
            out: ssa_out,
        }),
        1 => Ok(Op::Unary {
            name: op.to_string(),
            out: ssa_out,
            lhs: ssa_vars[0],
        }),
        2 => Ok(Op::Binary {
            name: op.to_string(),
            out: ssa_out,
            lhs: ssa_vars[0],
            rhs: ssa_vars[1],
        }),
        i => Err(format!("Cannot have {i} arguments (maximum of 2)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_lru() {
        let mut lru: Lru<2> = Lru::new();
        lru.poke(0);
        assert!(lru.pop() == 1);
        assert!(lru.pop() == 0);

        lru.poke(1);
        assert!(lru.pop() == 0);
        assert!(lru.pop() == 1);
    }

    #[test]
    fn test_medium_lru() {
        let mut lru: Lru<10> = Lru::new();
        lru.poke(0);
        for _ in 0..9 {
            assert!(lru.pop() != 0);
        }
        assert!(lru.pop() == 0);

        lru.poke(1);
        for _ in 0..9 {
            assert!(lru.pop() != 1);
        }
        assert!(lru.pop() == 1);

        lru.poke(4);
        lru.poke(5);
        for _ in 0..8 {
            assert!(!matches!(lru.pop(), 4 | 5));
        }
        assert!(lru.pop() == 4);
        assert!(lru.pop() == 5);
    }
}