use arrayvec::ArrayVec;
use std::collections::BTreeMap;
use std::fmt::Write;
use wasm_bindgen::prelude::*;

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct SsaVariable(usize);
impl SsaVariable {
    pub const INVALID: Self = Self(usize::MAX);
}

#[derive(Debug)]
enum Op<'a> {
    Binary {
        name: &'a str,
        out: SsaVariable,
        lhs: SsaVariable,
        rhs: SsaVariable,
    },
    Unary {
        name: &'a str,
        out: SsaVariable,
        lhs: SsaVariable,
    },
    Nonary {
        name: &'a str,
        out: SsaVariable,
    },
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Register(usize);
impl Register {
    pub const INVALID: Self = Self(usize::MAX);
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Memory(usize);

#[derive(Debug)]
enum AsmOp<'a> {
    Load {
        src: Memory,
        dst: Register,
    },
    Store {
        src: Register,
        dst: Memory,
    },
    Binary {
        name: &'a str,
        out: Register,
        lhs: Register,
        rhs: Register,
    },
    Unary {
        name: &'a str,
        out: Register,
        lhs: Register,
    },
    Nonary {
        name: &'a str,
        out: Register,
    },
}

impl AsmOp<'_> {
    fn to_string(&self, offset: usize) -> String {
        match self {
            AsmOp::Nonary { name, out } => format!("r{} = {name}()", out.0),
            AsmOp::Unary { name, out, lhs } => format!("r{} = {name}(r{})", out.0, lhs.0),
            AsmOp::Binary {
                name,
                out,
                lhs,
                rhs,
            } => format!("r{} = {name}(r{}, r{})", out.0, lhs.0, rhs.0),
            AsmOp::Load { src, dst } => format!("r{} = LOAD(mem + {})", dst.0, src.0 - offset),
            AsmOp::Store { src, dst } => format!("STORE(r{}, mem + {})", src.0, dst.0 - offset),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Allocation {
    Register(Register),
    Memory(Memory),
    Unassigned,
}

/// Cheap and cheerful single-pass register allocation
pub struct RegisterAllocator<'s, const N: usize> {
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
    out: Vec<AsmOp<'s>>,
}

impl<'s, const N: usize> RegisterAllocator<'s, N> {
    /// Builds a new `RegisterAllocator`.
    ///
    /// Upon construction, SSA register 0 is bound to local register 0; you
    /// would be well advised to use it as the output of your function.
    pub fn new() -> Self {
        Self {
            allocations: vec![],

            registers: [SsaVariable(usize::MAX); N],
            register_lru: Lru::new(),

            spare_registers: ArrayVec::new(),
            spare_memory: Vec::with_capacity(1024),

            total_slots: 1,
            out: Vec::with_capacity(1024),
        }
    }

    fn get_allocation(&mut self, i: SsaVariable) -> &mut Allocation {
        if i.0 >= self.allocations.len() {
            self.allocations.resize(i.0 + 1, Allocation::Unassigned);
        }
        if let Allocation::Register(r) = self.allocations[i.0] {
            self.register_lru.poke(r.0);
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

    fn bind_initial_register(&mut self, n: SsaVariable, reg: Register) {
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
        match *self.get_allocation(out) {
            Allocation::Register(r_x) => r_x,
            Allocation::Memory(m_x) => {
                // TODO: this could be more efficient with a Swap instruction,
                // since we know that we're about to free a memory slot.
                let r_a = self.get_register();

                self.out.push(AsmOp::Store { src: r_a, dst: m_x });
                self.release_mem(m_x);
                self.bind_register(out, r_a);
                r_a
            }
            Allocation::Unassigned => Register::INVALID,
        }
    }

    /// Lowers an operation that uses a single register into an
    /// [`AsmOp`](crate::asm::AsmOp), pushing it to the internal tape.
    ///
    /// This may also push `Load` or `Store` instructions to the internal tape,
    /// if there aren't enough spare registers.
    fn op_reg(&mut self, out: SsaVariable, arg: SsaVariable, name: &'s str) {
        let op = |out, lhs| AsmOp::Unary { out, lhs, name };
        let r_x = self.get_out_reg(out);
        if r_x == Register::INVALID {
            return;
        }
        match *self.get_allocation(arg) {
            Allocation::Register(r_y) => {
                assert!(r_x != r_y);
                self.out.push(op(r_x, r_y));
                self.release_reg(r_x);
            }
            Allocation::Memory(m_y) => {
                self.out.push(op(r_x, r_x));
                self.rebind_register(arg, r_x);

                self.out.push(AsmOp::Store { src: r_x, dst: m_y });
                self.release_mem(m_y);
            }
            Allocation::Unassigned => {
                self.out.push(op(r_x, r_x));
                self.bind_initial_register(arg, r_x);
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
    fn op_reg_reg(&mut self, out: SsaVariable, lhs: SsaVariable, rhs: SsaVariable, name: &'s str) {
        let op = |out, lhs, rhs| AsmOp::Binary {
            out,
            lhs,
            rhs,
            name,
        };
        let r_x = self.get_out_reg(out);
        if r_x == Register::INVALID {
            return;
        }
        match (*self.get_allocation(lhs), *self.get_allocation(rhs)) {
            (Allocation::Register(r_y), Allocation::Register(r_z)) => {
                self.out.push(op(r_x, r_y, r_z));
                self.release_reg(r_x);
            }
            (Allocation::Memory(m_y), Allocation::Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.rebind_register(lhs, r_x);

                self.out.push(AsmOp::Store { src: r_x, dst: m_y });
                self.release_mem(m_y);
            }
            (Allocation::Register(r_y), Allocation::Memory(m_z)) => {
                self.out.push(op(r_x, r_y, r_x));
                self.rebind_register(rhs, r_x);

                self.out.push(AsmOp::Store { src: r_x, dst: m_z });
                self.release_mem(m_z);
            }
            (Allocation::Memory(m_y), Allocation::Memory(m_z)) => {
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
            (Allocation::Unassigned, Allocation::Register(r_z)) => {
                self.out.push(op(r_x, r_x, r_z));
                self.bind_initial_register(lhs, r_x);
            }
            (Allocation::Register(r_y), Allocation::Unassigned) => {
                self.out.push(op(r_x, r_y, r_x));
                self.bind_initial_register(rhs, r_x);
            }
            (Allocation::Unassigned, Allocation::Unassigned) => {
                let r_a = if lhs == rhs { r_x } else { self.get_register() };

                self.out.push(op(r_x, r_x, r_a));
                self.bind_initial_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_initial_register(rhs, r_a);
                }
            }
            (Allocation::Unassigned, Allocation::Memory(m_z)) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_x, r_a));
                self.bind_initial_register(lhs, r_x);
                if lhs != rhs {
                    self.bind_register(rhs, r_a);
                }

                self.out.push(AsmOp::Store { src: r_a, dst: m_z });
                self.release_mem(m_z);
            }
            (Allocation::Memory(m_y), Allocation::Unassigned) => {
                let r_a = self.get_register();
                assert!(r_a != r_x);

                self.out.push(op(r_x, r_a, r_x));
                self.bind_register(lhs, r_a);
                assert!(lhs != rhs);
                self.bind_initial_register(rhs, r_x);

                self.out.push(AsmOp::Store { src: r_a, dst: m_y });
                self.release_mem(m_y);
            }
        }
    }

    fn op_out_only(&mut self, out: SsaVariable, name: &'s str) {
        let r_x = self.get_out_reg(out);
        if r_x == Register::INVALID {
            return;
        }
        self.out.push(AsmOp::Nonary { out: r_x, name });
        self.release_reg(r_x);
    }

    fn run(tape: &'s [Op]) -> Vec<AsmOp<'s>> {
        if tape.is_empty() {
            return vec![];
        }

        let mut alloc = Self::new();
        let out_ssa = match tape.last().unwrap() {
            Op::Nonary { out, .. } | Op::Unary { out, .. } | Op::Binary { out, .. } => *out,
        };
        alloc.bind_initial_register(out_ssa, Register(0));

        for t in tape.iter().rev() {
            match t {
                Op::Nonary { name, out } => alloc.op_out_only(*out, name),
                Op::Unary { name, out, lhs } => alloc.op_reg(*out, *lhs, name),
                Op::Binary {
                    name,
                    out,
                    lhs,
                    rhs,
                } => alloc.op_reg_reg(*out, *lhs, *rhs, name),
            }
        }
        alloc.out
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
///         ^                       oldest       newest     |
///         |-----------------------------------------------|
/// ```
pub struct Lru<const N: usize> {
    data: [LruNode; N],
    head: usize,
}

impl<const N: usize> Default for Lru<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Lru<N> {
    pub fn new() -> Self {
        let mut out = Self {
            head: 0,
            data: [LruNode { prev: 0, next: 0 }; N],
        };
        for i in 0..N {
            out.data[i].prev = i.checked_sub(1).unwrap_or(N - 1);
            out.data[i].next = (i + 1) % N;
        }

        out
    }

    /// Remove the given node from the list
    fn remove(&mut self, i: usize) {
        self.data[self.data[i].prev].next = self.data[i].next;
        self.data[self.data[i].next].prev = self.data[i].prev;
    }

    /// Inserts node `i` before location `next`
    fn insert_before(&mut self, i: usize, next: usize) {
        let prev = self.data[next].prev;
        self.data[prev].next = i;
        self.data[next].prev = i;
        self.data[i] = LruNode { next, prev };
    }

    /// Mark the given node as newest
    pub fn poke(&mut self, i: usize) {
        let prev_newest = self.head;
        if i == prev_newest {
            return;
        } else if self.data[prev_newest].prev != i {
            // If this wasn't the oldest node, then remove it and
            // reinsert itright before the head of the list.
            self.remove(i);
            self.insert_before(i, self.head);
        }
        self.head = i; // rotate
    }
    /// Look up the oldest node in the list, marking it as newest
    pub fn pop(&mut self) -> usize {
        let out = self.data[self.head].prev;
        self.head = out; // rotate
        out
    }
}

////////////////////////////////////////////////////////////////////////////////

#[wasm_bindgen(start)]
pub fn bind_console_err() {
    console_error_panic_hook::set_once();
}

fn do_the_thing_n<const N: usize>(s: String) -> String {
    if s.is_empty() {
        return "Error:\nInput is empty".to_owned();
    }
    match parse(&s) {
        Ok(s) => {
            let allocated = RegisterAllocator::<N>::run(&s);
            let mut out = "".to_owned();
            for s in allocated.into_iter().rev() {
                if !out.is_empty() {
                    out += "\n";
                }
                _ = write!(&mut out, "{}", s.to_string(2));
            }
            out
        }
        Err((line_num, err)) => format!("Error:\n{err}\n(line {})", line_num + 1),
    }
}

#[wasm_bindgen]
pub fn do_the_thing(s: String) -> String {
    do_the_thing_n::<2>(s)
}

#[wasm_bindgen]
pub fn do_the_thing_generic(s: String, n: usize) -> String {
    match n {
        2 => do_the_thing_n::<2>(s),
        3 => do_the_thing_n::<3>(s),
        4 => do_the_thing_n::<4>(s),
        5 => do_the_thing_n::<5>(s),
        6 => do_the_thing_n::<6>(s),
        7 => do_the_thing_n::<7>(s),
        8 => do_the_thing_n::<8>(s),
        9 => do_the_thing_n::<9>(s),
        10 => do_the_thing_n::<10>(s),
        11 => do_the_thing_n::<11>(s),
        12 => do_the_thing_n::<12>(s),
        13 => do_the_thing_n::<13>(s),
        14 => do_the_thing_n::<14>(s),
        15 => do_the_thing_n::<15>(s),
        16 => do_the_thing_n::<16>(s),
        17 => do_the_thing_n::<17>(s),
        18 => do_the_thing_n::<18>(s),
        19 => do_the_thing_n::<19>(s),
        20 => do_the_thing_n::<20>(s),
        21 => do_the_thing_n::<21>(s),
        22 => do_the_thing_n::<22>(s),
        23 => do_the_thing_n::<23>(s),
        24 => do_the_thing_n::<24>(s),
        25 => do_the_thing_n::<25>(s),
        26 => do_the_thing_n::<26>(s),
        27 => do_the_thing_n::<27>(s),
        28 => do_the_thing_n::<28>(s),
        29 => do_the_thing_n::<29>(s),
        30 => do_the_thing_n::<30>(s),
        31 => do_the_thing_n::<31>(s),
        32 => do_the_thing_n::<32>(s),
        _ => format!("Error:\nInvalid register count {} (expected 2-32)", n),
    }
}

fn parse(s: &str) -> Result<Vec<Op>, (usize, String)> {
    let mut vars = BTreeMap::new();
    let mut out = vec![];
    for (index, line) in s.split('\n').enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_line(line, &mut vars) {
            Ok(v) => out.push(v),
            Err(e) => return Err((index, e)),
        }
    }
    Ok(out)
}

fn parse_line<'a>(
    line: &'a str,
    vars: &mut BTreeMap<&'a str, SsaVariable>,
) -> Result<Op<'a>, String> {
    let line = line.trim();
    let mut iter = line.split('=');

    let out = iter.next().ok_or_else(|| "Missing =".to_owned())?.trim();
    if vars.contains_key(out) {
        return Err(format!("Duplicated assignment: {out}"));
    }
    let ssa_out = SsaVariable(vars.len());
    vars.insert(out, ssa_out);

    let op = iter
        .next()
        .ok_or_else(|| "Missing operation".to_owned())?
        .trim();
    let (op, rest) = op
        .split_once('(')
        .ok_or_else(|| "Missing opening (".to_owned())?;
    let op = op.trim();
    if op == "LOAD" || op == "STORE" {
        return Err(format!("Use of reserved function name: {}", op));
    }

    let rest = rest
        .trim_end_matches(';')
        .trim()
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
            name: op,
            out: ssa_out,
        }),
        1 => Ok(Op::Unary {
            name: op,
            out: ssa_out,
            lhs: ssa_vars[0],
        }),
        2 => Ok(Op::Binary {
            name: op,
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
