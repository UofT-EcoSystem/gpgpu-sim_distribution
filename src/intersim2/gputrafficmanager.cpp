// Copyright (c) 2009-2013, Tor M. Aamodt, Dongdong Li, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <sstream>
#include <fstream>
#include <limits> 

#include "gputrafficmanager.hpp"
#include "interconnect_interface.hpp"
#include "globals.hpp"


GPUTrafficManager::GPUTrafficManager( const Configuration &config, const vector<Network *> &net)
:TrafficManager(config, net)
{
  // The total simulations equal to number of kernels
  _total_sims = 0;
  
  _input_queue.resize(_subnets);
  for ( int subnet = 0; subnet < _subnets; ++subnet) {
    _input_queue[subnet].resize(_nodes);
    for ( int node = 0; node < _nodes; ++node ) {
      _input_queue[subnet][node].resize(_classes);
    }
  }

  _n_shader = config.GetInt("n_shader");
  _n_mem = config.GetInt("n_mem");
}

GPUTrafficManager::~GPUTrafficManager()
{
}

void GPUTrafficManager::Init()
{
  _time = 0;
  _sim_state = running;
  _ClearStats( );
  
}

void GPUTrafficManager::_RetireFlit( Flit *f, int dest, int subnet )
{
  _deadlock_timer = 0;
  
  assert(_total_in_flight_flits[f->cl].count(f->id) > 0);
  _total_in_flight_flits[f->cl].erase(f->id);
  
  if(f->record) {
    assert(_measured_in_flight_flits[f->cl].count(f->id) > 0);
    _measured_in_flight_flits[f->cl].erase(f->id);
  }
  
  if ( f->watch ) {
    *gWatchOut << GetSimTime() << " | "
    << "node" << dest << " | "
    << "Retiring flit " << f->id
    << " (packet " << f->pid
    << ", src = " << f->src
    << ", dest = " << f->dest
    << ", hops = " << f->hops
    << ", flat = " << f->atime - f->itime
    << ")." << endl;
  }
  
  if ( f->head && ( f->dest != dest ) ) {
    ostringstream err;
    err << "Flit " << f->id << " arrived at incorrect output " << dest;
    Error( err.str( ) );
  }
  
  if((_slowest_flit[subnet][f->cl] < 0) ||
     (_flat_stats[subnet][f->cl]->Max() < (f->atime - f->itime)))
    _slowest_flit[subnet][f->cl] = f->id;
  
  _flat_stats[subnet][f->cl]->AddSample( f->atime - f->itime);
  if(_pair_stats){
    _pair_flat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - f->itime );
  }
  
  if ( f->tail ) {
    Flit * head;
    if(f->head) {
      head = f;
    } else {
      map<int, Flit *>::iterator iter = _retired_packets[f->cl].find(f->pid);
      assert(iter != _retired_packets[f->cl].end());
      head = iter->second;
      _retired_packets[f->cl].erase(iter);
      assert(head->head);
      assert(f->pid == head->pid);
    }
    if ( f->watch ) {
      *gWatchOut << GetSimTime() << " | "
      << "node" << dest << " | "
      << "Retiring packet " << f->pid
      << " (plat = " << f->atime - head->ctime
      << ", nlat = " << f->atime - head->itime
      << ", frag = " << (f->atime - head->atime) - (f->id - head->id) // NB: In the spirit of solving problems using ugly hacks, we compute the packet length by taking advantage of the fact that the IDs of flits within a packet are contiguous.
      << ", src = " << head->src
      << ", dest = " << head->dest
      << ")." << endl;
    }
   
// GPGPUSim: Memory will handle reply, do not need this
#if 0
    //code the source of request, look carefully, its tricky ;)
    if (f->type == Flit::READ_REQUEST || f->type == Flit::WRITE_REQUEST) {
      PacketReplyInfo* rinfo = PacketReplyInfo::New();
      rinfo->source = f->src;
      rinfo->time = f->atime;
      rinfo->record = f->record;
      rinfo->type = f->type;
      _repliesPending[dest].push_back(rinfo);
    } else {
      if(f->type == Flit::READ_REPLY || f->type == Flit::WRITE_REPLY  ){
        _requestsOutstanding[dest]--;
      } else if(f->type == Flit::ANY_TYPE) {
        _requestsOutstanding[f->src]--;
      }
      
    }
#endif

    if(f->type == Flit::READ_REPLY || f->type == Flit::WRITE_REPLY  ){
      _requestsOutstanding[dest]--;
    } else if(f->type == Flit::ANY_TYPE) {
      ostringstream err;
      err << "Flit " << f->id << " cannot be ANY_TYPE" ;
      Error( err.str( ) );
    }
    
    // Only record statistics once per packet (at tail)
    // and based on the simulation state
    if ( ( _sim_state == warming_up ) || f->record ) {
      
      _hop_stats[subnet][f->cl]->AddSample( f->hops );
      
      if((_slowest_packet[subnet][f->cl] < 0) ||
         (_plat_stats[subnet][f->cl]->Max() < (f->atime - head->itime)))
        _slowest_packet[subnet][f->cl] = f->pid;
      _plat_stats[subnet][f->cl]->AddSample( f->atime - head->ctime);
      _nlat_stats[subnet][f->cl]->AddSample( f->atime - head->itime);
      _frag_stats[subnet][f->cl]->AddSample( (f->atime - head->atime) - (f->id - head->id) );
      
      if(_pair_stats){
        _pair_plat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - head->ctime );
        _pair_nlat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - head->itime );
      }
    }
    
    if(f != head) {
      head->Free();
    }
    
  }
  
  if(f->head && !f->tail) {
    _retired_packets[f->cl].insert(make_pair(f->pid, f));
  } else {
    f->Free();
  }
}
int  GPUTrafficManager::_IssuePacket( int source, int cl )
{
  return 0;
}

//TODO: Remove stype?
void GPUTrafficManager::_GeneratePacket(int source, int stype, int cl, int time, int subnet, int packet_size, const Flit::FlitType& packet_type, void* const data, int dest)
{
  assert(stype!=0);
  
  //  Flit::FlitType packet_type = Flit::ANY_TYPE;
  int size = packet_size; //input size
  int pid = _cur_pid++;
  assert(_cur_pid);
  int packet_destination = dest;
  bool record = false;
  bool watch = gWatchOut && (_packets_to_watch.count(pid) > 0);
  
  // In GPGPUSim, the core specified the packet_type and size
  
#if 0
  if(_use_read_write[cl]){
    if(stype > 0) {
      if (stype == 1) {
        packet_type = Flit::READ_REQUEST;
        size = _read_request_size[cl];
      } else if (stype == 2) {
        packet_type = Flit::WRITE_REQUEST;
        size = _write_request_size[cl];
      } else {
        ostringstream err;
        err << "Invalid packet type: " << packet_type;
        Error( err.str( ) );
      }
    } else {
      PacketReplyInfo* rinfo = _repliesPending[source].front();
      if (rinfo->type == Flit::READ_REQUEST) {//read reply
        size = _read_reply_size[cl];
        packet_type = Flit::READ_REPLY;
      } else if(rinfo->type == Flit::WRITE_REQUEST) {  //write reply
        size = _write_reply_size[cl];
        packet_type = Flit::WRITE_REPLY;
      } else {
        ostringstream err;
        err << "Invalid packet type: " << rinfo->type;
        Error( err.str( ) );
      }
      packet_destination = rinfo->source;
      time = rinfo->time;
      record = rinfo->record;
      _repliesPending[source].pop_front();
      rinfo->Free();
    }
  }
#endif
  
  if ((packet_destination <0) || (packet_destination >= _nodes)) {
    ostringstream err;
    err << "Incorrect packet destination " << packet_destination
    << " for stype " << packet_type;
    Error( err.str( ) );
  }
  
  if ( ( _sim_state == running ) ||
      ( ( _sim_state == draining ) && ( time < _drain_time ) ) ) {
    record = _measure_stats[cl];
  }
  
  int subnetwork = subnet;
  //                ((packet_type == Flit::ANY_TYPE) ?
  //                    RandomInt(_subnets-1) :
  //                    _subnet[packet_type]);
  
  if ( watch ) {
    *gWatchOut << GetSimTime() << " | "
    << "node" << source << " | "
    << "Enqueuing packet " << pid
    << " at time " << time
    << "." << endl;
  }
  
  for ( int i = 0; i < size; ++i ) {
    Flit * f  = Flit::New();
    f->id     = _cur_id++;
    assert(_cur_id);
    f->pid    = pid;
    f->watch  = watch | (gWatchOut && (_flits_to_watch.count(f->id) > 0));
    f->subnetwork = subnetwork;
    f->src    = source;
    f->ctime  = time;
    f->record = record;
    f->cl     = cl;
    f->data = data;
    
    _total_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    if(record) {
      _measured_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    }
    
    if(gTrace){
      cout<<"New Flit "<<f->src<<endl;
    }
    f->type = packet_type;
    
    if ( i == 0 ) { // Head flit
      f->head = true;
      //packets are only generated to nodes smaller or equal to limit
      f->dest = packet_destination;
    } else {
      f->head = false;
      f->dest = -1;
    }
    switch( _pri_type ) {
      case class_based:
        f->pri = _class_priority[cl];
        assert(f->pri >= 0);
        break;
      case age_based:
        f->pri = numeric_limits<int>::max() - time;
        assert(f->pri >= 0);
        break;
      case sequence_based:
        f->pri = numeric_limits<int>::max() - _packet_seq_no[source];
        assert(f->pri >= 0);
        break;
      default:
        f->pri = 0;
    }
    if ( i == ( size - 1 ) ) { // Tail flit
      f->tail = true;
    } else {
      f->tail = false;
    }
    
    f->vc  = -1;
    
    if ( f->watch ) {
      *gWatchOut << GetSimTime() << " | "
      << "node" << source << " | "
      << "Enqueuing flit " << f->id
      << " (packet " << f->pid
      << ") at time " << time
      << "." << endl;
    }
    
    _input_queue[subnet][source][cl].push_back( f );
  }
}

void GPUTrafficManager::_Step()
{
  bool flits_in_flight = false;
  for(int c = 0; c < _classes; ++c) {
    flits_in_flight |= !_total_in_flight_flits[c].empty();
  }
  if(flits_in_flight && (_deadlock_timer++ >= _deadlock_warn_timeout)){
    _deadlock_timer = 0;
    cout << "WARNING: Possible network deadlock.\n";
  }
  
  vector<map<int, Flit *> > flits(_subnets);
  
  for ( int subnet = 0; subnet < _subnets; ++subnet ) {
    for ( int n = 0; n < _nodes; ++n ) {

      /************* Handle network ejection ************/
      Flit * const f = _net[subnet]->ReadFlit( n );
      if ( f ) {
        if(f->watch) {
          *gWatchOut << GetSimTime() << " | "
          << "node" << n << " | "
          << "Ejecting flit " << f->id
          << " (packet " << f->pid << ")"
          << " from VC " << f->vc
          << "." << endl;
        }
        g_icnt_interface->WriteOutBuffer(subnet, n, f);
      }
      
      g_icnt_interface->Transfer2BoundaryBuffer(subnet, n);
      Flit* const ejected_flit = g_icnt_interface->GetEjectedFlit(subnet, n);
      if (ejected_flit) {
        if(ejected_flit->head)
          assert(ejected_flit->dest == n);
        if(ejected_flit->watch) {
          *gWatchOut << GetSimTime() << " | "
          << "node" << n << " | "
          << "Ejected flit " << ejected_flit->id
          << " (packet " << ejected_flit->pid
          << " VC " << ejected_flit->vc << ")"
          << "from ejection buffer." << endl;
        }
        flits[subnet].insert(make_pair(n, ejected_flit));
        if((_sim_state == warming_up) || (_sim_state == running)) {
          ++_accepted_flits[subnet][ejected_flit->cl][n];
          if(ejected_flit->tail) {
            ++_accepted_packets[subnet][ejected_flit->cl][n];
          }
        }
      }
    
      /************* Credit return ************/
      // Processing the credit From the network
      Credit * const c = _net[subnet]->ReadCredit( n );
      if ( c ) {
#ifdef TRACK_FLOWS
        for(set<int>::const_iterator iter = c->vc.begin(); iter != c->vc.end(); ++iter) {
          int const vc = *iter;
          assert(!_outstanding_classes[n][subnet][vc].empty());
          int cl = _outstanding_classes[n][subnet][vc].front();
          _outstanding_classes[n][subnet][vc].pop();
          assert(_outstanding_credits[cl][subnet][n] > 0);
          --_outstanding_credits[cl][subnet][n];
        }
#endif
        _buf_states[n][subnet]->ProcessCredit(c);
        c->Free();
      }
    }

    /************* Handle inputs to network  ************/
    _net[subnet]->ReadInputs( );
  }

// GPGPUSim will generate/inject packets from interconnection interface
#if 0
  if ( !_empty_network ) {
    _Inject();
  }
#endif
  
  for(int subnet = 0; subnet < _subnets; ++subnet) {
    
    for(int n = 0; n < _nodes; ++n) {
      
      Flit * f = NULL;
      
      BufferState * const dest_buf = _buf_states[n][subnet];
      
      int const last_class = _last_class[n][subnet];
      
      int class_limit = _classes;
      
      if(_hold_switch_for_packet) {
        list<Flit *> const & pp = _input_queue[subnet][n][last_class];
        if(!pp.empty() && !pp.front()->head &&
           !dest_buf->IsFullFor(pp.front()->vc)) {
          f = pp.front();
          assert(f->vc == _last_vc[n][subnet][last_class]);
          
          // if we're holding the connection, we don't need to check that class
          // again in the for loop
          --class_limit;
        }
      }
      
      for(int i = 1; i <= class_limit; ++i) {
        
        int const c = (last_class + i) % _classes;
        
        list<Flit *> const & pp = _input_queue[subnet][n][c];
        
        if(pp.empty()) {
          continue;
        }
        
        Flit * const cf = pp.front();
        assert(cf);
        assert(cf->cl == c);
        
        assert(cf->subnetwork == subnet);
        
        if(f && (f->pri >= cf->pri)) {
          continue;
        }
        
        if(cf->head && cf->vc == -1) { // Find first available VC
          
          OutputSet route_set;
          _rf(NULL, cf, -1, &route_set, true);
          set<OutputSet::sSetElement> const & os = route_set.GetSet();
          assert(os.size() == 1);
          OutputSet::sSetElement const & se = *os.begin();
          assert(se.output_port == -1);
          int vc_start = se.vc_start;
          int vc_end = se.vc_end;
          int vc_count = vc_end - vc_start + 1;
          if(_noq) {
            assert(_lookahead_routing);
            const FlitChannel * inject = _net[subnet]->GetInject(n);
            const Router * router = inject->GetSink();
            assert(router);
            int in_channel = inject->GetSinkPort();
            
            // NOTE: Because the lookahead is not for injection, but for the
            // first hop, we have to temporarily set cf's VC to be non-negative
            // in order to avoid seting of an assertion in the routing function.
            cf->vc = vc_start;
            _rf(router, cf, in_channel, &cf->la_route_set, false);
            cf->vc = -1;
            
            if(cf->watch) {
              *gWatchOut << GetSimTime() << " | "
              << "node" << n << " | "
              << "Generating lookahead routing info for flit " << cf->id
              << " (NOQ)." << endl;
            }
            set<OutputSet::sSetElement> const sl = cf->la_route_set.GetSet();
            assert(sl.size() == 1);
            int next_output = sl.begin()->output_port;
            vc_count /= router->NumOutputs();
            vc_start += next_output * vc_count;
            vc_end = vc_start + vc_count - 1;
            assert(vc_start >= se.vc_start && vc_start <= se.vc_end);
            assert(vc_end >= se.vc_start && vc_end <= se.vc_end);
            assert(vc_start <= vc_end);
          }
          if(cf->watch) {
            *gWatchOut << GetSimTime() << " | " << FullName() << " | "
            << "Finding output VC for flit " << cf->id
            << ":" << endl;
          }
          for(int i = 1; i <= vc_count; ++i) {
            int const lvc = _last_vc[n][subnet][c];
            int const vc =
            (lvc < vc_start || lvc > vc_end) ?
            vc_start :
            (vc_start + (lvc - vc_start + i) % vc_count);
            assert((vc >= vc_start) && (vc <= vc_end));
            if(!dest_buf->IsAvailableFor(vc)) {
              if(cf->watch) {
                *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                << "  Output VC " << vc << " is busy." << endl;
              }
            } else {
              if(dest_buf->IsFullFor(vc)) {
                if(cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                  << "  Output VC " << vc << " is full." << endl;
                }
              } else {
                if(cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                  << "  Selected output VC " << vc << "." << endl;
                }
                cf->vc = vc;
                break;
              }
            }
          }
        }
        
        if(cf->vc == -1) {
          if(cf->watch) {
            *gWatchOut << GetSimTime() << " | " << FullName() << " | "
            << "No output VC found for flit " << cf->id
            << "." << endl;
          }
        } else {
          if(dest_buf->IsFullFor(cf->vc)) {
            if(cf->watch) {
              *gWatchOut << GetSimTime() << " | " << FullName() << " | "
              << "Selected output VC " << cf->vc
              << " is full for flit " << cf->id
              << "." << endl;
            }
          } else {
            f = cf;
          }
        }
      }
      
      if(f) {
        
        assert(f->subnetwork == subnet);
        
        int const c = f->cl;
        
        if(f->head) {
          
          if (_lookahead_routing) {
            if(!_noq) {
              const FlitChannel * inject = _net[subnet]->GetInject(n);
              const Router * router = inject->GetSink();
              assert(router);
              int in_channel = inject->GetSinkPort();
              _rf(router, f, in_channel, &f->la_route_set, false);
              if(f->watch) {
                *gWatchOut << GetSimTime() << " | "
                << "node" << n << " | "
                << "Generating lookahead routing info for flit " << f->id
                << "." << endl;
              }
            } else if(f->watch) {
              *gWatchOut << GetSimTime() << " | "
              << "node" << n << " | "
              << "Already generated lookahead routing info for flit " << f->id
              << " (NOQ)." << endl;
            }
          } else {
            f->la_route_set.Clear();
          }
          
          dest_buf->TakeBuffer(f->vc);
          _last_vc[n][subnet][c] = f->vc;
        }
        
        _last_class[n][subnet] = c;
        
        _input_queue[subnet][n][c].pop_front();
        
#ifdef TRACK_FLOWS
        ++_outstanding_credits[c][subnet][n];
        _outstanding_classes[n][subnet][f->vc].push(c);
#endif
        
        dest_buf->SendingFlit(f);
        
        if(_pri_type == network_age_based) {
          f->pri = numeric_limits<int>::max() - _time;
          assert(f->pri >= 0);
        }
        
        if(f->watch) {
          *gWatchOut << GetSimTime() << " | "
          << "node" << n << " | "
          << "Injecting flit " << f->id
          << " into subnet " << subnet
          << " at time " << _time
          << " with priority " << f->pri
          << "." << endl;
        }
        f->itime = _time;
        
        // Pass VC "back"
        if(!_input_queue[subnet][n][c].empty() && !f->tail) {
          Flit * const nf = _input_queue[subnet][n][c].front();
          nf->vc = f->vc;
        }
        
        if((_sim_state == warming_up) || (_sim_state == running)) {
          ++_sent_flits[subnet][c][n];
          if(f->head) {
            ++_sent_packets[subnet][c][n];
          }
        }
        
#ifdef TRACK_FLOWS
        ++_injected_flits[c][n];
#endif
        
        _net[subnet]->WriteFlit(f, n);
        
      }
    }
  }
  //Send the credit To the network
  for(int subnet = 0; subnet < _subnets; ++subnet) {
    for(int n = 0; n < _nodes; ++n) {
      map<int, Flit *>::const_iterator iter = flits[subnet].find(n);
      if(iter != flits[subnet].end()) {
        Flit * const f = iter->second;

        f->atime = _time;
        if(f->watch) {
          *gWatchOut << GetSimTime() << " | "
          << "node" << n << " | "
          << "Injecting credit for VC " << f->vc
          << " into subnet " << subnet
          << "." << endl;
        }
        Credit * const c = Credit::New();
        c->vc.insert(f->vc);
        _net[subnet]->WriteCredit(c, n);
        
#ifdef TRACK_FLOWS
        ++_ejected_flits[f->cl][n];
#endif
        
        _RetireFlit(f, n, subnet);
      }
    }
    flits[subnet].clear();
    // _InteralStep here
    _net[subnet]->Evaluate( );
    _net[subnet]->WriteOutputs( );
  }
  
  ++_time;
  assert(_time);
  if(gTrace){
    cout<<"TIME "<<_time<<endl;
  }
  
}

void GPUTrafficManager::DisplayStats(ostream & os) const {

    for (int subnet = 0; subnet < _subnets; subnet++) {
        unsigned send_node_start, send_node_count, rcv_node_start, rcv_node_count;

        if (subnet == 0) {
            send_node_start = 0;
            send_node_count = _n_shader;
            rcv_node_start = _n_shader;
            rcv_node_count = _n_mem;
        } else {
            send_node_start = _n_shader;
            send_node_count = _n_mem;
            rcv_node_start = 0;
            rcv_node_count = _n_shader;
        }

        for(int c = 0; c < _classes; ++c) {

            if(_measure_stats[c] == 0) {
                continue;
            }

            cout << "------ Subnet " << subnet << ", " << "Class " << c << ": ------" << endl;

            string index = "[" + to_string(subnet) + "," + to_string(c) + "]";

            cout
            << index << "Packet latency average = " << _plat_stats[subnet][c]->Average() << endl
            << "\tminimum = " << _plat_stats[subnet][c]->Min() << endl
            << "\tmaximum = " << _plat_stats[subnet][c]->Max() << endl
            << index << "Network latency average = " << _nlat_stats[subnet][c]->Average() << endl
            << "\tminimum = " << _nlat_stats[subnet][c]->Min() << endl
            << "\tmaximum = " << _nlat_stats[subnet][c]->Max() << endl
            << index << "Slowest packet = " << _slowest_packet[subnet][c] << endl
            << index << "Flit latency average = " << _flat_stats[subnet][c]->Average() << endl
            << "\tminimum = " << _flat_stats[subnet][c]->Min() << endl
            << "\tmaximum = " << _flat_stats[subnet][c]->Max() << endl
            << index << "Slowest flit = " << _slowest_flit[subnet][c] << endl
            << index << "Fragmentation average = " << _frag_stats[subnet][c]->Average() << endl
            << "\tminimum = " << _frag_stats[subnet][c]->Min() << endl
            << "\tmaximum = " << _frag_stats[subnet][c]->Max() << endl;

            int count_sum, count_min, count_max;
            double rate_sum, rate_min, rate_max;
            double rate_avg;
            int sent_packets, sent_flits, accepted_packets, accepted_flits;
            int min_pos, max_pos;
            double time_delta = (double)(_time - _reset_time);
            _ComputeStatsSubVector(_sent_packets[subnet][c], send_node_start, send_node_count, &count_sum, &count_min, &count_max, &min_pos, &max_pos);
            rate_sum = (double)count_sum / time_delta;
            rate_min = (double)count_min / time_delta;
            rate_max = (double)count_max / time_delta;
            rate_avg = rate_sum / (double)send_node_count;
            sent_packets = count_sum;
            cout << index << "Injected packet rate average = " << rate_avg << endl
                    << "\tminimum = " << rate_min
                    << " (at node " << min_pos << ")" << endl
                    << "\tmaximum = " << rate_max
                    << " (at node " << max_pos << ")" << endl;
            _ComputeStatsSubVector(_accepted_packets[subnet][c], rcv_node_start, rcv_node_count, &count_sum, &count_min, &count_max, &min_pos, &max_pos);
            rate_sum = (double)count_sum / time_delta;
            rate_min = (double)count_min / time_delta;
            rate_max = (double)count_max / time_delta;
            rate_avg = rate_sum / (double)rcv_node_count;
            accepted_packets = count_sum;
            cout << index << "Accepted packet rate average = " << rate_avg << endl
                    << "\tminimum = " << rate_min
                    << " (at node " << min_pos << ")" << endl
                    << "\tmaximum = " << rate_max
                    << " (at node " << max_pos << ")" << endl;
            _ComputeStatsSubVector(_sent_flits[subnet][c], send_node_start, send_node_count, &count_sum, &count_min, &count_max, &min_pos, &max_pos);
            rate_sum = (double)count_sum / time_delta;
            rate_min = (double)count_min / time_delta;
            rate_max = (double)count_max / time_delta;
            rate_avg = rate_sum / (double)send_node_count;
            sent_flits = count_sum;
            cout << index << "Injected flit rate average = " << rate_avg << endl
                    << "\tminimum = " << rate_min
                    << " (at node " << min_pos << ")" << endl
                    << "\tmaximum = " << rate_max
                    << " (at node " << max_pos << ")" << endl;
            _ComputeStatsSubVector(_accepted_flits[subnet][c], rcv_node_start, rcv_node_count, &count_sum, &count_min, &count_max, &min_pos, &max_pos);
            rate_sum = (double)count_sum / time_delta;
            rate_min = (double)count_min / time_delta;
            rate_max = (double)count_max / time_delta;
            rate_avg = rate_sum / (double)rcv_node_count;
            accepted_flits = count_sum;
            cout << index << "Accepted flit rate average= " << rate_avg << endl
                    << "\tminimum = " << rate_min
                    << " (at node " << min_pos << ")" << endl
                    << "\tmaximum = " << rate_max
                    << " (at node " << max_pos << ")" << endl;

            cout << index << "Injected packet length average = " << (double)sent_flits / (double)sent_packets << endl
                    << "Accepted packet length average = " << (double)accepted_flits / (double)accepted_packets << endl;


#ifdef TRACK_STALLS
            _ComputeStats(_buffer_busy_stalls[c], &count_sum);
            rate_sum = (double)count_sum / time_delta;
            rate_avg = rate_sum / (double)(_subnets*_routers);
            os << "Buffer busy stall rate = " << rate_avg << endl;
            _ComputeStats(_buffer_conflict_stalls[c], &count_sum);
            rate_sum = (double)count_sum / time_delta;
            rate_avg = rate_sum / (double)(_subnets*_routers);
            os << "Buffer conflict stall rate = " << rate_avg << endl;
            _ComputeStats(_buffer_full_stalls[c], &count_sum);
            rate_sum = (double)count_sum / time_delta;
            rate_avg = rate_sum / (double)(_subnets*_routers);
            os << "Buffer full stall rate = " << rate_avg << endl;
            _ComputeStats(_buffer_reserved_stalls[c], &count_sum);
            rate_sum = (double)count_sum / time_delta;
            rate_avg = rate_sum / (double)(_subnets*_routers);
            os << "Buffer reserved stall rate = " << rate_avg << endl;
            _ComputeStats(_crossbar_conflict_stalls[c], &count_sum);
            rate_sum = (double)count_sum / time_delta;
            rate_avg = rate_sum / (double)(_subnets*_routers);
            os << "Crossbar conflict stall rate = " << rate_avg << endl;
#endif

        }
    }

    cout << "------In-flight flits------" << endl;

    for (int c = 0; c < _classes; c++) {
        cout << "class(" << c << ")" << "Total in-flight flits = " << _total_in_flight_flits[c].size()
             << " (" << _measured_in_flight_flits[c].size() << " measured)"
             << endl;
    }
}

void GPUTrafficManager::_ComputeStatsSubVector( const vector<int> & stats, const int left, const int count,
        int *sum, int *min, int *max, int *min_pos, int *max_pos ) const
{
    assert(count > 0);
    assert((left < stats.size()) && (left + count <= stats.size()));

    if(min_pos) {
        *min_pos = left;
    }
    if(max_pos) {
        *max_pos = left;
    }

    if(min) {
        *min = stats[left];
    }
    if(max) {
        *max = stats[left];
    }

    *sum = stats[left];

    for ( int i = 1; i < count; ++i ) {
        int curr = stats[left + i];
        if ( min  && ( curr < *min ) ) {
            *min = curr;
            if ( min_pos ) {
                *min_pos = left + i;
            }
        }
        if ( max && ( curr > *max ) ) {
            *max = curr;
            if ( max_pos ) {
                *max_pos = left + i;
            }
        }
        *sum += curr;
    }
}
